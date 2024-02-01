import torch
from torch import nn
import torch.distributed

from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.input_metadata import InputMetadata

from llama_llm import LlamaDecodeAttention, KVCache

class TensorParallelDecodeLlamaAttention(LlamaDecodeAttention):

    def __init__(self, **kargs) -> None:
        super().__init__(**kargs)

        # Overwrite values for TP
        self.num_heads = self.num_heads // self.n_gpus
        self.num_kv_heads = self.num_kv_heads // self.n_gpus

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = nn.Linear(self.hidden_size, self.q_size + 2 * self.kv_size,
                                  device=self.device, dtype=self.dtype)
        self.o_proj = nn.Linear(self.hidden_size // self.n_gpus, self.hidden_size,
                                device=self.device, dtype=self.dtype)

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
        output = super().forward(hidden_states, kv_cache, input_metadata)

        # All-reduce on output
        torch.distributed.all_reduce(output)

        return output