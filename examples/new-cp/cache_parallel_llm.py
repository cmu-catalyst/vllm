from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed
from torch import nn

from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.input_metadata import InputMetadata
from vllm._C import ops
from vllm.utils import get_max_shared_memory_bytes
from vllm.model_executor.input_metadata import InputMetadata
from vllm._C import cache_ops

from llama_llm import LlamaDecoderLayer, LlamaDecodeAttention, KVCache

_PARTITION_SIZE = 512

def _distributed_paged_attention(
    rank: int,
    world_size: int,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    input_metadata: InputMetadata,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
) -> torch.Tensor:
    # batch_size = len(input_metadata.context_lens)
    # query = query[:batch_size, :, :]
    n_rank0_batch_size = input_metadata.n_rank0_batch_size

    block_size = value_cache.shape[3]
    num_seqs, num_heads, head_size = query.shape

    # print(f"(rank, n_seqs, n_heads, head_size) = ({rank}, {num_seqs}, {num_heads}, {head_size})")
    max_num_partitions = (
            (input_metadata.max_context_len + _PARTITION_SIZE - 1) //
            _PARTITION_SIZE)
    assert _PARTITION_SIZE % block_size == 0
    output = torch.empty_like(query)
    tmp_output = torch.empty(
        size=(num_seqs, num_heads, max_num_partitions, head_size),
        dtype=output.dtype,
        device=output.device,
    )
    exp_sums = torch.empty(
        size=(num_seqs, num_heads, max_num_partitions),
        dtype=torch.float32,
        device=output.device,
    )
    max_logits = torch.empty_like(exp_sums)

    ops.paged_attention_v2(
        output,
        exp_sums,
        max_logits,
        tmp_output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        input_metadata.block_tables,
        input_metadata.context_lens,
        block_size,
        input_metadata.max_context_len,
        alibi_slopes,
    )

    local_output = None
    if rank > 0:
        local_output = output[n_rank0_batch_size:,:,:]
        # print("(rank, local_output_shape) = ", rank, local_output.shape)

    exp_sums = exp_sums[:n_rank0_batch_size, :, 0].contiguous()
    max_logits = max_logits[:n_rank0_batch_size, :, 0].contiguous()
    output = output[:n_rank0_batch_size, :, :]

    exp_sums_list, max_logits_list, output_list = None, None, None
    if rank == 0: # Receive from other GPUs
        exp_sums_list = [torch.empty_like(exp_sums) for _ in range(world_size)]
        max_logits_list = [torch.empty_like(max_logits) for _ in range(world_size)]
        output_list = [torch.empty_like(output) for _ in range(world_size)] #  [num_seqs, num_heads, head_size]

    torch.distributed.gather(exp_sums, gather_list=exp_sums_list, dst=0)
    torch.distributed.gather(max_logits, gather_list=max_logits_list, dst=0)
    torch.distributed.gather(output, gather_list=output_list, dst=0)

    # if rank == 0:
    #     print("rank, exp_sum, max_logits, output: ", rank, exp_sums_list[0].shape, max_logits_list[0].shape, output_list[0].shape)
    # else:
    #     print("rank, exp_sum, max_logits, output: ", rank, exp_sums_list, max_logits_list, output_list)

    if rank == 0:
        max_logits_list = torch.stack(max_logits_list, dim=-1)
        exp_sums_list = torch.stack(exp_sums_list, dim=-1)
        output_list = torch.stack(output_list, dim=-2)

        global_output = torch.zeros_like(output)
        ops.distributed_paged_attention_v2_reduce(
            global_output,
            exp_sums_list,
            max_logits_list,
            output_list,
            world_size
        )

    if rank > 0:
        global_output = local_output
    # print("rank, global output shape: ", rank, global_output.shape)

    return global_output

    # num_seqs_per_rank = num_seqs // world_size
    # seq_idx_st =  rank * num_seqs_per_rank
    # seq_idx_end = seq_idx_st + num_seqs_per_rank
    #
    # exp_sums = exp_sums[seq_idx_st:seq_idx_end, :, 0].contiguous()
    # max_logits = max_logits[seq_idx_st:seq_idx_end, :, 0].contiguous()
    # output = output[seq_idx_st:seq_idx_end, :, :]
    #
    # exp_sums_list = [torch.empty_like(exp_sums) for _ in range(world_size)]
    # max_logits_list = [torch.empty_like(max_logits) for _ in range(world_size)]
    # output_list = [torch.empty_like(output) for _ in range(world_size)] #  [num_seqs, num_heads, head_size]
    #
    # # Note(Soo): Reduce results
    # torch.distributed.all_gather(max_logits_list, max_logits)
    # torch.distributed.all_gather(exp_sums_list, exp_sums)
    # torch.distributed.all_gather(output_list, output)
    #
    # max_logits_list = torch.stack(max_logits_list, dim=-1)
    # exp_sums_list = torch.stack(exp_sums_list, dim=-1)
    # output_list = torch.stack(output_list, dim=-2)
    #
    # global_output = torch.zeros_like(output)
    # ops.distributed_paged_attention_v2_reduce(
    #     global_output,
    #     exp_sums_list,
    #     max_logits_list,
    #     output_list,
    #     world_size
    # )


    return global_output

class DistributedPagedAttention(nn.Module):
    def __init__(
        self,
        cp_size: int,
        num_heads: int,
        head_size: int,
        scale: float,
        rank: int,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cp_size = cp_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.alibi_slopes = alibi_slopes
        self.rank = rank

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata
    ) -> torch.Tensor:
        """PagedAttention forward pass.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for the inputs.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # FIXME(Soo): Enable adding new k and v to KV cache (+ slot_mapping)
        # assert input_metadata.slot_mapping is not None
        # # Reshape the keys and values and store them in the cache.
        # # If key_cache and value_cache are not provided, the new key and value
        # # vectors will not be cached. This happens during the initial memory
        # # profiling run.
        # if key_cache is not None and value_cache is not None:
        #     cache_ops.reshape_and_cache(
        #         key,
        #         value,
        #         key_cache,
        #         value_cache,
        #         input_metadata.slot_mapping.flatten(),
        #     )

        # Decoding run.
        output = _distributed_paged_attention(
            self.rank,
            self.cp_size,
            query,
            key_cache,
            value_cache,
            input_metadata,
            self.num_kv_heads,
            self.scale,
            self.alibi_slopes,
        )

        # Reshape the output tensor.
        return output.view(output.shape[0], seq_len, hidden_size)


class CacheParallelDecodeLlamaAttention(LlamaDecodeAttention):

    def __init__(self, **kargs) -> None:
        super().__init__(**kargs)

        # FIXME(Soo): Keep new KV cache to only one of ranks
        self.attn = DistributedPagedAttention(
            cp_size=self.n_gpus,
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            rank=self.rank
        )
        self.do_broadcast = True

    def forward(
        self,
        # positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        # NOTE(Soo): Replicate an activation (thus a new query to all devices)
        output = super().forward(hidden_states, kv_cache, input_metadata)

        return output

class CacheParallelDecodeLlamaLayer(LlamaDecoderLayer):

    def __init__(self, **kargs) -> None:
        super().__init__(**kargs)

        del kargs["intermediate_size"]
        self.self_attn = CacheParallelDecodeLlamaAttention(**kargs)

    def forward(
        self,
        # positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:

        output = super().forward(hidden_states, kv_cache, input_metadata)

        return output