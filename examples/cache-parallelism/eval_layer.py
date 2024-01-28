
import os
import torch

from transformers import LlamaConfig
from cache_parallel_llm import CacheParallelDecodeLlamaAttention
from synthetic_data_utils import gen_random_kv_cache, gen_model_input_metadata

def run_local_model(rank, world_size):
    # Evaluation config
    seed = 0
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dtype = torch.bfloat16
    # FIXME(Soo): Extend it to multi-gpu settings
    device = torch.device(f"cuda:{rank}")

    # Input / Model configs
    num_seqs = 1
    # FIXME(Soo): Push context_len to the boundary for long context setup
    # e.g., 100K
    max_kv_cache_context_len = 2000
    # TODO(Soo): Test with varying KV cache context length
    context_lens = [max_kv_cache_context_len for _ in range(num_seqs)]
    context_lens = torch.tensor(context_lens, dtype=torch.int, device=device)
    decode_context_len = 1
    kv_cache_len = 128
    num_layers = 1

    # Load Llama-7B configurations for testing
    llama_cfg = LlamaConfig()
    hidden_size = llama_cfg.hidden_size
    num_heads = llama_cfg.num_attention_heads
    num_kv_heads = llama_cfg.num_key_value_heads
    head_size = hidden_size // num_heads
    rope_theta = getattr(llama_cfg, "rope_theta", 10000)
    rope_scaling = getattr(llama_cfg, "rope_scaling", None)
    max_position_embeddings = getattr(llama_cfg, "max_position_embeddings", 8192)

    # KV cache block configs
    # FIXME(Soo): Push num_blocks to the boundary for long context setup
    num_blocks = 30000
    block_size = 16
    partition_size = 512

    attn = CacheParallelDecodeLlamaAttention(
        cp_size=world_size,
        device=device,
        dtype=dtype,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        max_position_embeddings=max_position_embeddings,
    )

    # Initialize hidden states
    scale = float(1.0 / (head_size ** 0.5))
    hidden_states = torch.empty(num_seqs, decode_context_len, hidden_size,
                                dtype=dtype, device=device)
    hidden_states.uniform_(-scale, scale)

    kv_cache = gen_random_kv_cache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        seed=seed,
        device=device,
    )
    input_metadata = gen_model_input_metadata(
        num_seqs, context_lens, max_kv_cache_context_len, block_size, device)

    # TODO(Soo): Increase KV cache size over iterations
    output = attn(hidden_states, kv_cache, input_metadata)
    print(f"Rank: {rank} / {world_size} done - output shape: {output.shape}")


def run_distributed_model(
        rank,
        world_size,
        _,
):
    # Set up the distributed environment
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.device(rank)

    # Run models
    run_local_model(rank, world_size)

    # Clean up
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # Evaluation config
    n_gpus = 4
    dummy = -1
    torch.multiprocessing.spawn(run_distributed_model,
                                args=(n_gpus, dummy),
                                nprocs=n_gpus,
                                join=True)

# Measure time
# Creating start and end events
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)
# start_event.record()
#
#
# end_event.record()
# torch.cuda.synchronize()
#
# # Calculating elapsed time
# elapsed_time_ms = start_event.elapsed_time(end_event)
# print(f"Elapsed time: {elapsed_time_ms} ms")
