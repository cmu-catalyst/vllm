from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]

class LlamaDecodeAttention(nn.Module):

    def __init__(
        self,
        n_gpus: int,
        rank: int,
        device: torch.device,
        dtype: torch.dtype,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
    ) -> None:
        super().__init__()
        torch.cuda.device(device)

        self.device = device
        self.dtype = dtype

        self.hidden_size = hidden_size
        self.n_gpus = n_gpus
        self.rank = rank
        # tp_size = get_tensor_model_parallel_world_size()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % self.num_heads == 0, "This should hold to work"
        self.head_dim = hidden_size // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings


        # NOTE(Soo): Replicate an activation (thus a new query to all devices)
        # > All projections (QKV, Output) are equivalent to the data-parallel case
        self.qkv_proj = nn.Linear(self.hidden_size, self.q_size + 2 * self.kv_size,
                                  device=device, dtype=dtype)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size,
                                device=device, dtype=dtype)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.do_broadcast = False

    def forward(
        self,
        # positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        # Replicated qkv projection across GPUs
        qkv = self.qkv_proj(hidden_states)

        if self.do_broadcast:
            # HACK(Soo): Assume rank 0 is always the one dealing with long sequence that requires CP
            # print(self.rank, qkv.shape)
            if self.rank == 0:
                qkv_rank0 = qkv
                # print("prepare attn (rank 0): ", qkv_rank0.shape)
            else:
                qkv_rank0 = torch.zeros([input_metadata.n_rank0_batch_size, qkv.shape[1], qkv.shape[2]],
                                        device=self.device, dtype=self.dtype)
                # print("prepare attn: ", self.rank, qkv_rank0.shape)

            torch.distributed.broadcast(qkv_rank0, src=0)

            if self.rank > 0:
                qkv = torch.cat([qkv_rank0, qkv], dim=0)
            # print("after merge: ", self.rank, qkv.shape)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # FIXME(Soo): Enable rotary embedding
        # q, k = self.rotary_emb(positions, q, k)

        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output = self.o_proj(attn_output)

        return output

class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.ffn1 = nn.Linear(hidden_size, intermediate_size*2, device=device, dtype=dtype)
        self.ffn2 = nn.Linear(intermediate_size, hidden_size, device=device, dtype=dtype)

        self.act_fn = SiluAndMul()

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act_fn(x)
        x = self.ffn2(x)
        return x

class LlamaDecoderLayer(nn.Module):

    def __init__(self, **kargs) -> None:
        super().__init__()

        self.mlp = LlamaMLP(
            hidden_size=kargs["hidden_size"],
            intermediate_size=kargs["intermediate_size"],
            device=kargs["device"],
            dtype=kargs["dtype"],
        )
        # self.input_layernorm = RMSNorm(config.hidden_size,
        #                                eps=config.rms_norm_eps)
        # self.post_attention_layernorm = RMSNorm(config.hidden_size,
        #                                         eps=config.rms_norm_eps)

    def forward(
        self,
        # positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        # residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        # if residual is None:
        #     residual = hidden_states
        #     hidden_states = self.input_layernorm(hidden_states)
        # else:
        #     hidden_states, residual = self.input_layernorm(
        #         hidden_states, residual)
        hidden_states = self.self_attn(
            # positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )

        # Fully Connected
        # hidden_states, residual = self.post_attention_layernorm(
        #     hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states #, residual

class LlamaModel(nn.Module):
    def __init__(self, **kargs) -> None:
        super().__init__()
        # self.config = config
        # self.padding_idx = config.pad_token_id
        # self.vocab_size = config.vocab_size
        # self.embed_tokens = VocabParallelEmbedding(
        #     config.vocab_size,
        #     config.hidden_size,
        # )
        layer_cls = kargs["layer_cls"]
        num_layers = kargs["num_layers"]
        del kargs["layer_cls"]
        del kargs["num_layers"]
        self.layers = nn.ModuleList([
            layer_cls(**kargs) for _ in range(num_layers)
        ])
        # self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        # input_ids: torch.Tensor,
        # positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        # hidden_states = self.embed_tokens(input_ids)
        # residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(hidden_states, kv_caches[i], input_metadata)
            # hidden_states, residual = layer(
            #     positions,
            #     hidden_states,
            #     kv_caches[i],
            #     input_metadata,
            #     residual,
            # )
        # hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
