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

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
# - Shared memory here is per CUDA block (that executes on a single SM)
# MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
MAX_SEQ_LEN = 1000 # To boost unit test, it is set lower than original MAX
NUM_BLOCKS = 5000  # Arbitrary values for testing
PARTITION_SIZE = 512 # Used for parallel KV cache load in FlashInfer

DTYPES = [torch.bfloat16] #, torch.float]
NUM_GEN_SEQS = [2, 4]  # Arbitrary values for testing
NUM_HEADS = [(32, 32)]  # LLaMA-7B
HEAD_SIZES = [128] # LLaMA-7B
BLOCK_SIZES = [16]
USE_ALIBI = [False]
SEEDS = [0]

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

def distributed_run(
        rank, world_size, q, key_cache, value_cache, context_lens, scale,
        block_tables, block_size, max_context_len, alibi_slopes, num_seqs, ref_output):
    # Set up the distributed environment
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # TODO(Soo): Implement cache-parallel distributed decode Llama attention
    # print(f"Rank: {rank} / {world_size}")
    gpu_id = f"cuda:{rank}"
    q = q.to(gpu_id)

    output = torch.empty_like(q)
    max_context_len = max_context_len // world_size
    num_partitions = ((max_context_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
    num_seqs, num_heads, head_size = output.shape
    tmp_output = torch.empty(
        size=(num_seqs, num_heads, num_partitions, head_size),
        dtype=output.dtype,
        device=output.device,
    )
    exp_sums = torch.empty(
        size=(num_seqs, num_heads, num_partitions),
        dtype=torch.float32,
        device=output.device,
    )
    max_logits = torch.empty_like(exp_sums)
    num_kv_heads = num_heads
    context_lens = context_lens // world_size

    # NOTE(Soo): Update KV cache (push k, v from i + context_lens_2 to i)
    k_cache = key_cache.to(gpu_id)
    v_cache = value_cache.to(gpu_id)
    block_tables = block_tables.to(gpu_id)
    context_lens = context_lens.to(gpu_id)
    for i in range(num_seqs):
        block_table = block_tables[i]
        context_len = int(context_lens[i])
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            idx_2 = (j + rank * context_len)
            block_number_2 = int(block_table[idx_2 // block_size])
            block_offset_2 = idx_2 % block_size
            k_cache[block_number, :, :, block_offset, :] = k_cache[block_number_2, :, :, block_offset_2, :]
            v_cache[block_number, :, :, block_offset] = v_cache[block_number_2, :, :, block_offset_2]

    ops.paged_attention_v2(
        output,
        exp_sums,
        max_logits,
        tmp_output,
        q,
        k_cache,
        v_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
    )
    exp_sums = exp_sums[:,:,0].contiguous()
    max_logits = max_logits[:,:,0].contiguous()
    exp_sums_list = [torch.empty_like(exp_sums) for _ in range(world_size)]
    max_logits_list = [torch.empty_like(max_logits) for _ in range(world_size)]
    output_list = [torch.empty_like(output) for _ in range(world_size)]

    # # TODO(Soo): Reduce results
    torch.distributed.all_gather(max_logits_list, max_logits)
    torch.distributed.all_gather(exp_sums_list, exp_sums)
    torch.distributed.all_gather(output_list, output)
    global_output = torch.zeros_like(output)
    for i in range(num_seqs):
        for j in range(num_heads):
            global_max_logit = max_logits_list[0][i][j]
            global_exp_sum = exp_sums_list[0][i][j]
            global_output[i][j] = output_list[0][i][j]
            for dev_id in range(1, world_size):
                prev_global_max_logit = global_max_logit
                max_logit_val = max_logits_list[dev_id][i][j]
                global_max_logit = max(global_max_logit, max_logit_val)

                new_global_exp_sum = global_exp_sum * math.exp(prev_global_max_logit - global_max_logit)
                global_exp_sum = new_global_exp_sum.clone()
                local_exp_sum = exp_sums_list[dev_id][i][j] * math.exp(max_logit_val - global_max_logit)
                global_exp_sum += local_exp_sum
                global_output[i][j] = global_output[i][j] * new_global_exp_sum / global_exp_sum
                global_output[i][j] += output_list[dev_id][i][j] * local_exp_sum / global_exp_sum

    ref_output = ref_output.to(gpu_id)
    assert torch.allclose(global_output, ref_output, atol=1e-3, rtol=1e-5)

    # Clean up
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("version", ["v2"])
@pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("use_alibi", USE_ALIBI)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_distributed_paged_attention(
    kv_cache_factory,
    version: str,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
) -> None:

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # TODO(Soo): Generate q, k, v, kv_cache, input_metadata
    gpu_id = f"cuda:0"
    scale = float(1.0 / (head_size ** 0.5))
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

    context_lens = [4 * random.randint(1, MAX_SEQ_LEN // 4) for _ in range(num_seqs)]
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
    # TODO(Soo): Do reference attention
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
        alibi_slopes
    )

    # TODO(Soo): Implement distributed attention
    # print("")
    world_size = 4
    torch.multiprocessing.spawn(distributed_run,
                                args=(world_size,
                                      query,
                                      key_cache,
                                      value_cache,
                                      context_lens,
                                      scale,
                                      block_tables,
                                      block_size,
                                      max_context_len,
                                      alibi_slopes,
                                      num_seqs,
                                      ref_output),
                                nprocs=world_size,
                                join=True)
    # print("Distributed attention compute done")