import copy
import os
import torch
import numpy as np
from typing import Any, Dict, List

from transformers import LlamaConfig
from cache_parallel_llm import CacheParallelDecodeLlamaAttention
from data_parallel_llm import DataParallelDecodeLlamaAttention
from synthetic_data_utils import gen_random_kv_cache, gen_model_input_metadata
from eval_logger import EvaluationLogger, EvaluationConfig

def get_model_by_parallelism_type(
        p_type: str,
        model_args: Dict[str, Any],
):
    attn = None
    device = model_args["device"]

    if p_type == "cp":
        model_args["cp_size"] = model_args["n_gpus"]
        del model_args["n_gpus"]
        attn = CacheParallelDecodeLlamaAttention(**model_args)
        attn = attn.to(device)
    elif p_type == "dp":
        model_args["dp_size"] = model_args["n_gpus"]
        del model_args["n_gpus"]
        attn = DataParallelDecodeLlamaAttention(**model_args)
        attn = attn.to(device)
    elif p_type == "tp":
        raise NotImplementedError("TP is not ready")
    else:
        raise ValueError("Invalid parallelism type")

    return attn

def adjust_configs_by_parallelism_type(
        cfg: EvaluationConfig,
        context_lens: List[int],
        rank: int,
):
    p_type = cfg.p_type
    if p_type == "cp":
        context_lens = context_lens // cfg.n_gpus
    elif p_type == "dp":
        cfg.num_seqs = cfg.num_seqs // cfg.n_gpus

        start_idx = rank * cfg.num_seqs
        end_idx = rank * cfg.num_seqs + cfg.num_seqs
        context_lens = context_lens[start_idx:end_idx]
    elif p_type == "tp":
        raise NotImplementedError("TP is not ready")
    else:
        raise ValueError("Invalid parallelism type")

    return cfg, context_lens

@torch.inference_mode()
def run_local_model(
        rank: int,
        cfg: EvaluationConfig,
):
    # Evaluation config
    torch.random.manual_seed(cfg.rand_seed)
    torch.cuda.manual_seed(cfg.rand_seed)
    device = torch.device(f"cuda:{rank}")

    # Input / Model configs
    # TODO(Soo): Test with varying KV cache context length
    context_lens = [cfg.max_kv_cache_context_len for _ in range(cfg.num_seqs)]
    context_lens = torch.tensor(context_lens, dtype=torch.int, device=device)
    decode_context_len = 1

    # Load Llama-7B configurations for testing
    llama_cfg = cfg.llama_cfg
    hidden_size = llama_cfg.hidden_size
    num_heads = llama_cfg.num_attention_heads
    num_kv_heads = llama_cfg.num_key_value_heads
    head_size = hidden_size // num_heads
    rope_theta = getattr(llama_cfg, "rope_theta", 10000)
    rope_scaling = getattr(llama_cfg, "rope_scaling", None)
    max_position_embeddings = getattr(llama_cfg, "max_position_embeddings", 8192)

    cfg, context_lens = adjust_configs_by_parallelism_type(
        cfg=cfg,
        context_lens=context_lens,
        rank=rank,
    )
    cfg.max_kv_cache_context_len = max(context_lens.cpu().numpy())

    model_args = {
        "n_gpus": cfg.n_gpus,
        "device": device,
        "dtype": cfg.dtype,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "rope_theta": rope_theta,
        "rope_scaling": rope_scaling,
        "max_position_embeddings": max_position_embeddings,
    }

    attn = get_model_by_parallelism_type(cfg.p_type, model_args)

    # Initialize hidden states
    scale = float(1.0 / (head_size ** 0.5))
    hidden_states = torch.empty(cfg.num_seqs, decode_context_len, hidden_size,
                                dtype=cfg.dtype, device=device)
    hidden_states.uniform_(-scale, scale)

    kv_cache = gen_random_kv_cache(
        num_blocks=cfg.num_blocks,
        block_size=cfg.block_size,
        num_layers=cfg.num_layers,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=cfg.dtype,
        seed=cfg.rand_seed,
        device=device,
    )
    input_metadata = gen_model_input_metadata(
        cfg.num_seqs,
        context_lens,
        cfg.max_kv_cache_context_len,
        cfg.block_size,
        device
    )

    # TODO(Soo): Increase KV cache size over iterations
    arr_elapsed_time_ms = []
    for _ in range(cfg.n_warmup_iters + cfg.n_eval_iters):
        # Measure time
        # Creating start and end events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output = attn(hidden_states, kv_cache, input_metadata)

        end_event.record()
        torch.cuda.synchronize()

        # Calculating elapsed time
        elapsed_time_ms = start_event.elapsed_time(end_event)
        arr_elapsed_time_ms.append(elapsed_time_ms)
        # print(f"Rank: {rank} / {world_size} done - output shape: {output.shape}")
        # print(f"Elapsed time ({rank}): {elapsed_time_ms} ms")

    arr_elapsed_time_ms = arr_elapsed_time_ms[cfg.n_warmup_iters:]
    avg_elapsed_time_ms = np.mean(arr_elapsed_time_ms)
    print(f"Avg Elapsed time ({rank}): {avg_elapsed_time_ms} ms")

    return avg_elapsed_time_ms

def save_result_to_file(
        cfg: EvaluationConfig,
        measurement: float,
):
    eval_logger = EvaluationLogger(cfg.output_file_path)

    # Add a evaluation data point
    eval_logger.add_result(
        model=cfg.model_name,
        batch_size=cfg.num_seqs,
        seq_len=cfg.max_kv_cache_context_len,
        p_type=cfg.p_type,
        measurement=measurement,
    )

    # Saving the results to a file
    eval_logger.save_results()

def run_distributed_model(
        rank: int,
        eval_cfg: EvaluationConfig,
):
    # Set up the distributed environment
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl",
                                         rank=rank,
                                         world_size=eval_cfg.n_gpus)
    torch.cuda.device(rank)
    init_cfg = copy.deepcopy(eval_cfg)

    # Run models
    avg_latency_ms = run_local_model(rank, eval_cfg)

    # Log results
    if rank == 0:
        # Use init_cfg instead of eval_cfg since eval_cfg will change
        # depending on parallelism type
        save_result_to_file(init_cfg, avg_latency_ms)

    # Clean up
    torch.distributed.destroy_process_group()


def check_eval_configs(
    cfg: EvaluationConfig,
):
    # FIXME(Soo): Enable imbalanced situation for all parallelism types
    # It currently assumes ideal situation for all parallleisms
    assert cfg.max_kv_cache_context_len % cfg.n_gpus == 0 # CP
    assert cfg.num_seqs % cfg.n_gpus == 0 # DP
    num_heads = cfg.llama_cfg.num_attention_heads
    assert num_heads % cfg.n_gpus == 0  # TP

if __name__ == "__main__":
    # Evaluation config
    eval_cfg = EvaluationConfig(
        # Evaluation configs
        n_gpus = 4,
        n_eval_iters = 10,
        n_warmup_iters = 3,
        output_file_path = "/home/byungsoj/eval_results/result.json",
        rand_seed = 0,
        p_type="cp",

        # Model configs
        dtype = torch.bfloat16,
        model_name = "Llama-7B",
        num_seqs = 4,
        # FIXME(Soo): Push context_len to the boundary for long context setup
        max_kv_cache_context_len = 2000,
        num_layers = 1,
        llama_cfg = LlamaConfig(), # Load Llama-7B configurations

        # vLLM configs
        # FIXME(Soo): Push num_blocks to the boundary for long context setup
        num_blocks = 30000,
        block_size = 16,
        partition_size = 512,
    )

    p_types = ["cp"]
    # p_types = ["dp"]
    max_kv_cache_context_lens = [8000, 16000]
    for p_type in p_types:
        for cache_len in max_kv_cache_context_lens:
            eval_cfg.p_type = p_type
            eval_cfg.max_kv_cache_context_len = cache_len
            check_eval_configs(eval_cfg)
            torch.multiprocessing.spawn(run_distributed_model,
                                        args=(eval_cfg,),
                                        nprocs=eval_cfg.n_gpus,
                                        join=True)