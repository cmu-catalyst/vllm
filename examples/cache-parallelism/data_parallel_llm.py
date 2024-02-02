import torch

from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.input_metadata import InputMetadata

from llama_llm import LlamaDecoderLayer, LlamaDecodeAttention, KVCache

class DataParallelDecodeLlamaAttention(LlamaDecodeAttention):

    def __init__(self, **kargs) -> None:
        super().__init__(**kargs)

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

        return output

class DataParallelDecodeLlamaLayer(LlamaDecoderLayer):

    def __init__(self, **kargs) -> None:
        super().__init__(**kargs)

        del kargs["intermediate_size"]
        self.self_attn = DataParallelDecodeLlamaAttention(**kargs)

    def forward(
        self,
        # positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:

        output = super().forward(hidden_states, kv_cache, input_metadata)

        return output