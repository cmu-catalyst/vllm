from typing import List, Tuple

import torch
from vllm.model_executor.input_metadata import InputMetadata

def create_kv_caches(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape,
                                dtype=dtype,
                                device=device)
        key_cache.uniform_(-scale, scale)
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape,
                                  dtype=dtype,
                                  device=device)
        value_cache.uniform_(-scale, scale)
        value_caches.append(value_cache)
    return key_caches, value_caches

def gen_random_kv_cache(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int,
    device: torch.device,
):
    # TODO(Soo): Create kv cache
    key_caches, value_caches = create_kv_caches(
        num_blocks, block_size, num_layers, num_kv_heads, head_size, dtype, seed, device)

    kv_caches = []
    for k_cache, v_cache in zip(key_caches, value_caches):
        kv_caches.append((k_cache, v_cache))

    return kv_caches

def gen_block_table_and_slot_mapping(num_seqs, context_lens, block_size, device):
    # FIXME(Soo): Later, adjust to make it work for increasing KV cache size over iters
    # We need to make the best use of memory footprint by key, value cache without
    # idle space in the middle; still, it wouldn't matter much to performance
    # since CUDA kernel reads each token at a time from physical address
    # by block_table anyway although consecutive ones would give better physical locality
    block_tables = []
    total_num_blocks = 0
    for i in range(num_seqs):
        max_num_blocks_per_seq = (context_lens[i] + block_size - 1) // block_size
        block_table = [
            total_num_blocks + j
            for j in range(max_num_blocks_per_seq)
        ]
        total_num_blocks += max_num_blocks_per_seq
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device=device)

    # FIXME(Soo): Enable slot_mapping if needed later
    return block_tables, None


def gen_model_input_metadata(num_seqs, context_lens, max_context_len, block_size, device):
    block_tables, slot_mapping = gen_block_table_and_slot_mapping(
        num_seqs, context_lens, block_size, device)

    input_metadata = InputMetadata(
        is_prompt=False,
        slot_mapping=slot_mapping,
        max_context_len=max_context_len,
        context_lens=context_lens,
        block_tables=block_tables,
        use_cuda_graph=False,
    )

    return input_metadata


