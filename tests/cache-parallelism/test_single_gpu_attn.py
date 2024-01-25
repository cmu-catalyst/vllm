import math
import random
from typing import List, Optional, Tuple

import pytest
import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from vllm._C import ops
from vllm.utils import get_max_shared_memory_bytes

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
# - Shared memory here is per CUDA block (that executes on a single SM)
#
# With RTX A5000 on the Catalyst cluster
# - max_shared_memory_bytes = ~97KB
# - MAX_SEQ_LEN = 24832
# - With 40000 GPU blocks, it will be 24832 * 4 bytes * 40000 (blocks) / 2^30 = ~3.7GB
# MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
MAX_SEQ_LEN = 10000 # To boost unit test, it is set lower than original MAX
NUM_BLOCKS = 30000  # Arbitrary values for testing
PARTITION_SIZE = 512 # Used for parallel KV cache load in FlashInfer

DTYPES = [torch.half, torch.bfloat16] #, torch.float]
NUM_GEN_SEQS = [5]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
NUM_HEADS = [(40, 40)]  # LLaMA-13B
HEAD_SIZES = [128] # LLaMA-13B
BLOCK_SIZES = [16]
USE_ALIBI = [False]
SEEDS = [0]
DEVICES = [i for i in range(1 if torch.cuda.device_count() == 1 else 2)]

def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def ref_single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables = block_tables.cpu().tolist()
    context_lens = context_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(context_len, device=query.device).int()
            alibi_bias = (position_ids - context_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(
                1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)


@pytest.mark.parametrize("version", ["v2"])
@pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("use_alibi", USE_ALIBI)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
def test_paged_attention(
    kv_cache_factory,
    version: str,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    device: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    gpu_id = f"cuda:{device}"
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs,
                        num_query_heads,
                        head_size,
                        dtype=dtype,
                        device=gpu_id)
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads,
                                   dtype=torch.float,
                                   device=gpu_id)

    context_lens = [2 * random.randint(1, MAX_SEQ_LEN // 2) for _ in range(num_seqs)]
    context_lens[-1] = MAX_SEQ_LEN
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device=gpu_id)

    # Create the block tables.
    # NOTE(Soo): Set block table in a way that there is no hash collision
    # This is needed to avoid overwriting key or values when we move
    # second half of the sequence to first half for testing
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for i in range(num_seqs):
        block_table = [
            # random.randint(0, NUM_BLOCKS - 1)
            i * max_num_blocks_per_seq + j
            for j in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device=gpu_id)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(NUM_BLOCKS, block_size, 1,
                                                num_kv_heads, head_size, dtype,
                                                seed, gpu_id)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Run the reference implementation before updating KV cache
    ref_output = torch.empty_like(query)
    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        num_queries_per_kv,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        scale,
        alibi_slopes,
    )

    # NOTE(Soo): Call two separate "Distributed" attention and reduce it
    # First attention computes first half of sequence
    assert version == "v2"
    assert PARTITION_SIZE % block_size == 0

    output_1 = torch.empty_like(query)
    max_context_len_1 = max_context_len // 2
    num_partitions_1 = ((max_context_len_1 + PARTITION_SIZE - 1) //
                      PARTITION_SIZE)

    num_seqs, num_heads, head_size = output_1.shape
    tmp_output_1 = torch.empty(
        size=(num_seqs, num_heads, num_partitions_1, head_size),
        dtype=output_1.dtype,
        device=output_1.device,
    )
    exp_sums_1 = torch.empty(
        size=(num_seqs, num_heads, num_partitions_1),
        dtype=torch.float32,
        device=output_1.device,
    )
    max_logits_1 = torch.empty_like(exp_sums_1)
    context_lens_1 = context_lens // 2
    ops.paged_attention_v2(
        output_1,
        exp_sums_1,
        max_logits_1,
        tmp_output_1,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens_1,
        block_size,
        max_context_len_1,
        alibi_slopes,
    )

    # NOTE(Soo): Second "Distributed" attention computing second half of sequence
    output_2 = torch.empty_like(query)
    max_context_len_2 = max_context_len // 2
    num_partitions_2 = ((max_context_len_2 + PARTITION_SIZE - 1) //
                        PARTITION_SIZE)

    num_seqs, num_heads, head_size = output_2.shape
    tmp_output_2 = torch.empty(
        size=(num_seqs, num_heads, num_partitions_2, head_size),
        dtype=output_2.dtype,
        device=output_2.device,
    )
    exp_sums_2 = torch.empty(
        size=(num_seqs, num_heads, num_partitions_2),
        dtype=torch.float32,
        device=output_2.device,
    )
    max_logits_2 = torch.empty_like(exp_sums_2)
    context_lens_2 = context_lens // 2

    # NOTE(Soo): Update KV cache (push k, v from i + context_lens_2 to i)
    for i in range(num_seqs):
        block_table = block_tables[i]
        context_len = int(context_lens_2[i])
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            idx_2 = (j + context_len)
            block_number_2 = int(block_table[idx_2 // block_size])
            block_offset_2 = idx_2 % block_size
            key_cache[block_number, :, :, block_offset, :] = key_cache[block_number_2, :, :, block_offset_2, :]
            value_cache[block_number, :, :, block_offset] = value_cache[block_number_2, :, :, block_offset_2]

    ops.paged_attention_v2(
        output_2,
        exp_sums_2,
        max_logits_2,
        tmp_output_2,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens_2,
        block_size,
        max_context_len_2,
        alibi_slopes,
    )

    # TODO(Soo): Reduce two "Distributed" attentions to verify max_logits / exp_sums
    exp_sums_1 = exp_sums_1[:,:,0].contiguous()
    exp_sums_2 = exp_sums_2[:, :, 0].contiguous()
    max_logits_1 = max_logits_1[:, :, 0].contiguous()
    max_logits_2 = max_logits_2[:, :, 0].contiguous()
    global_max_logits = torch.empty(
        size=(num_seqs, num_heads),
        dtype=torch.float32,
        device=output_2.device,
    )
    global_exp_sum = torch.empty_like(global_max_logits)
    global_output = torch.empty_like(query)
    for i in range(num_seqs):
        for j in range(num_heads):
            global_max_logits[i][j] = max(max_logits_1[i][j], max_logits_2[i][j])
            new_exp_sum_1 = exp_sums_1[i][j] * math.exp(max_logits_1[i][j] - global_max_logits[i][j])
            new_exp_sum_2 = exp_sums_2[i][j] * math.exp(max_logits_2[i][j] - global_max_logits[i][j])
            global_exp_sum[i][j] = new_exp_sum_1 + new_exp_sum_2
            global_output[i][j] = output_1[i][j] * new_exp_sum_1 / global_exp_sum[i][j]
            global_output[i][j] += output_2[i][j] * new_exp_sum_2 / global_exp_sum[i][j]

    # NOTE(woosuk): Due to the kernel-level differences in the two
    # implementations, there is a small numerical difference in the two
    # outputs. Thus, we use a relaxed tolerance for the test.
    assert torch.allclose(global_output, ref_output, atol=1e-3, rtol=1e-5)