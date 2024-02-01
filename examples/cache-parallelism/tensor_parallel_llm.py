import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from vllm._C import ops
from vllm.utils import get_max_shared_memory_bytes

import torch.distributed
from torch.distributed import ReduceOp
from torch import nn

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

from vllm._C import cache_ops

KVCache = Tuple[torch.Tensor, torch.Tensor]
_PARTITION_SIZE = 512

class TensorParallelDecodeLlamaAttention(nn.Module):

    def __init__(
        self,
        tp_size: int, # degree of cache parallelism
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

        self.hidden_size = hidden_size
        self.tp_size = tp_size
        # tp_size = get_tensor_model_parallel_world_size()

        self.num_heads = num_heads // self.tp_size
        self.num_kv_heads = num_kv_heads // self.tp_size
        assert hidden_size % self.num_heads == 0, "This should hold to work"
        self.head_dim = hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # NOTE(Soo): Replicate an activation (thus a new query to all devices)
        # > All projections (QKV, Output) are equivalent to the data-parallel case
        self.qkv_proj = nn.Linear(self.hidden_size, self.q_size + 2 * self.kv_size,
                                  device=device, dtype=dtype)
        self.o_proj = nn.Linear(self.hidden_size // self.tp_size, self.hidden_size,
                                device=device, dtype=dtype)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        # TODO(Soo): Implement distributed Paged Attention
        self.attn = PagedAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(
        self,
        # positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        # Replicated qkv projection across GPUs
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # FIXME(Soo): Enable rotary embedding
        # q, k = self.rotary_emb(positions, q, k)

        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output = self.o_proj(attn_outpu

        # All-reduce on output
        torch.distributed.all_reduce(output)

        return output