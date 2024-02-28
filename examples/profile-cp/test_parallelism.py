import copy
import os
import time
import torch
import numpy as np
import random
from typing import Any, Dict, List, Tuple

from transformers import LlamaConfig
from llama_llm import LlamaModel
from cache_parallel_llm import CacheParallelDecodeLlamaAttention, CacheParallelDecodeLlamaLayer
from data_parallel_llm import DataParallelDecodeLlamaAttention, DataParallelDecodeLlamaLayer
from tensor_parallel_llm import TensorParallelDecodeLlamaAttention, TensorParallelDecodeLlamaLayer
from synthetic_data_utils import gen_random_kv_cache
from eval_logger import EvaluationLogger, EvaluationConfig
from kv_cache_manager import BatchManager

def get_model_by_parallelism_type(
        p_type: str,
        model_args: Dict[str, Any],
):
    model = None
    device = model_args["device"]

    p_type_to_layer_cls = {
        "cp": CacheParallelDecodeLlamaLayer,
        "dp": DataParallelDecodeLlamaLayer,
        "tp": TensorParallelDecodeLlamaLayer
    }
    if p_type not in p_type_to_layer_cls:
        raise ValueError("Invalid parallelism type")
    model_args["layer_cls"] = p_type_to_layer_cls[p_type]

    model = LlamaModel(**model_args)
    model = model.to(device)

    return model

def get_layer_by_parallelism_type(
        p_type: str,
        model_args: Dict[str, Any],
):
    model = None
    device = model_args["device"]

    if p_type == "cp":
        model = CacheParallelDecodeLlamaLayer(**model_args)
    elif p_type == "dp":
        model = DataParallelDecodeLlamaLayer(**model_args)
    elif p_type == "tp":
        model = TensorParallelDecodeLlamaLayer(**model_args)
    else:
        raise ValueError("Invalid parallelism type")

    model = model.to(device)
    return model

def get_attn_by_parallelism_type(
        p_type: str,
        model_args: Dict[str, Any],
):
    attn = None
    device = model_args["device"]

    if p_type == "cp":
        attn = CacheParallelDecodeLlamaAttention(**model_args)
    elif p_type == "dp":
        attn = DataParallelDecodeLlamaAttention(**model_args)
    elif p_type == "tp":
        attn = TensorParallelDecodeLlamaAttention(**model_args)
    else:
        raise ValueError("Invalid parallelism type")

    attn = attn.to(device)
    return attn

def adjust_configs_by_parallelism_type(
        cfg: EvaluationConfig,
        context_lens: List[int],
        target_context_lens: List[int],
        rank: int,
):
    p_type = cfg.p_type
    if p_type == "cp":
        context_lens = context_lens // cfg.n_gpus
        target_context_lens = [i // cfg.n_gpus for i in target_context_lens]
    elif p_type == "dp":
        num_seqs_per_gpu = cfg.num_seqs // cfg.n_gpus

        start_idx = rank * num_seqs_per_gpu
        end_idx = rank * num_seqs_per_gpu + num_seqs_per_gpu
        context_lens = context_lens[start_idx:end_idx]
        target_context_lens = target_context_lens[start_idx:end_idx]
    elif p_type == "tp":
        pass
    else:
        raise ValueError("Invalid parallelism type")

    # NOTE(Soo): Check if it goes beyond our KV cache block size
    # total_tokens = sum(context_lens)
    # tp_divisor = cfg.n_gpus if p_type == "tp" else 1
    # assert total_tokens / tp_divisor <= cfg.num_blocks * cfg.block_size, "KV cache OOM"

    return cfg, context_lens, target_context_lens

def init_context_lens(
        max_kv_cache_context_len: int,
        n_min_decode_iters: int,
        n_max_decode_iters: int,
        num_seqs: int,
        n_gpus: int,
        device: torch.device,
        long_seq_ratio: float = 0.2,
        min_seq_len_multiplier: float = 0.2,
        min_seq_len: int = 10,
):
    min_kv_cache_context_len = max(int(min_seq_len_multiplier * max_kv_cache_context_len), min_seq_len)
    context_lens = [max_kv_cache_context_len for _ in range(num_seqs)]
    target_context_lens = [max_kv_cache_context_len + n_max_decode_iters for _ in range(num_seqs)]
    n_long_seqs = int(num_seqs * long_seq_ratio)

    for i, l in enumerate(context_lens):
        if i >= n_long_seqs:
            start_len = min_kv_cache_context_len
            end_len = min(2 * min_kv_cache_context_len, max_kv_cache_context_len)
            context_lens[i] = (random.randint(start_len, end_len) // n_gpus) * n_gpus

            n_decode_iters = (random.randint(n_min_decode_iters, n_max_decode_iters) // n_gpus) * n_gpus
            target_context_lens[i] = context_lens[i] + n_decode_iters

    # Fair load balancing for DP (in case of long prefix case)
    # fairness_factor = 0.2
    # n_swap_long_seqs = int(n_long_seqs * fairness_factor)
    # num_seqs_per_gpu = num_seqs // n_gpus
    # for i in range(n_swap_long_seqs):
    #     new_idx = (i % n_gpus) * num_seqs_per_gpu + (i // n_gpus)
    #     context_lens[i], context_lens[new_idx] = context_lens[new_idx], context_lens[i]
    #     target_context_lens[i], target_context_lens[new_idx] = target_context_lens[new_idx], target_context_lens[i]

    # DEBUG(Soo): For debugging
    # context_lens = [max_kv_cache_context_len for _ in range(num_seqs)]
    # target_context_lens = [max_kv_cache_context_len + 50 for _ in range(num_seqs)]

    return context_lens, target_context_lens


@torch.inference_mode()
def run_local_model(
        rank: int,
        cfg: EvaluationConfig,
):
    # Evaluation config
    torch.random.manual_seed(cfg.rand_seed)
    torch.cuda.manual_seed(cfg.rand_seed)
    random.seed(cfg.rand_seed)
    device = torch.device(f"cuda:{rank}")

    # Input / Model configs
    # TODO(Soo): Test with varying KV cache context length
    context_lens, target_context_lens = init_context_lens(
        cfg.max_kv_cache_context_len, cfg.n_min_decode_iters, cfg.n_max_decode_iters,
        cfg.num_seqs, cfg.n_gpus, device)
    n_total_tokens = sum([t - c for (c, t) in zip(context_lens, target_context_lens)])
    context_lens = torch.tensor(context_lens, dtype=torch.int, device=device)

    # Load Llama-7B configurations for testing
    llama_cfg = cfg.llama_cfg
    hidden_size = llama_cfg.hidden_size
    num_heads = llama_cfg.num_attention_heads
    num_kv_heads = llama_cfg.num_key_value_heads
    head_size = hidden_size // num_heads

    rope_theta = getattr(llama_cfg, "rope_theta", 10000)
    rope_scaling = getattr(llama_cfg, "rope_scaling", None)
    max_position_embeddings = getattr(llama_cfg, "max_position_embeddings", 8192)

    cfg, context_lens, target_context_lens = adjust_configs_by_parallelism_type(
        cfg=cfg,
        context_lens=context_lens,
        target_context_lens=target_context_lens,
        rank=rank,
    )
    cfg.max_kv_cache_context_len = max(context_lens.cpu().numpy())

    model_args = {
        "n_gpus": cfg.n_gpus,
        "rank": rank,
        "device": device,
        "dtype": cfg.dtype,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "rope_theta": rope_theta,
        "rope_scaling": rope_scaling,
        "max_position_embeddings": max_position_embeddings,
    }

    # assert cfg.num_layers == 1
    # model = get_attn_by_parallelism_type(cfg.p_type, model_args)

    assert cfg.num_layers == 1
    model_args["intermediate_size"] = llama_cfg.intermediate_size
    model = get_layer_by_parallelism_type(cfg.p_type, model_args)

    # model_args["intermediate_size"] = llama_cfg.intermediate_size
    # model_args["num_layers"] = cfg.num_layers
    # model = get_model_by_parallelism_type(cfg.p_type, model_args)


    # Initialize hidden states
    scale = float(1.0 / (head_size ** 0.5))

    kv_caches = gen_random_kv_cache(
        num_blocks=cfg.num_blocks,
        block_size=cfg.block_size,
        num_layers=cfg.num_layers,
        num_kv_heads=num_kv_heads if cfg.p_type != "tp" else num_kv_heads // cfg.n_gpus,
        head_size=head_size,
        dtype=cfg.dtype,
        seed=cfg.rand_seed,
        device=device,
    )

    # Note(Soo): Hack
    if not isinstance(model, LlamaModel):
        kv_caches = kv_caches[0]

    # Generate CPU KV cache
    cpu_key_cache = torch.empty(kv_caches[0].shape, dtype=cfg.dtype, device='cpu', pin_memory=True)
    cpu_value_cache = torch.empty(kv_caches[1].shape, dtype=cfg.dtype, device='cpu', pin_memory=True)
    cpu_key_cache.uniform_(-1, 1)
    cpu_value_cache.uniform_(-1, 1)
    cpu_kv_caches = (cpu_key_cache, cpu_value_cache)

    # TODO(Soo): Increase KV cache size over iterations
    # print("con len: ", context_lens)
    # print("target con len: ", target_context_lens)
    arr_elapsed_time_ms = []
    for _ in range(cfg.n_warmup_iters + cfg.n_eval_iters):
        # Initialize batch manager
        batch_manager = BatchManager(
            cfg, rank, device, hidden_size, context_lens, target_context_lens, kv_caches, cpu_kv_caches)

        # Measure time
        # Creating start and end events
        start_time = time.perf_counter()
        iter_id = 1
        while batch_manager.is_running():
            hidden_states = batch_manager.gen_hidden_states(scale)
            input_metadata = batch_manager.gen_input_metadata()

            output = model(hidden_states, kv_caches, input_metadata)

            batch_manager.update(iter_id)
            # if rank == 0:
            #     print(f"Iter {iter_id}: {len(batch_manager.wait_queue)}, {len(batch_manager.running_queue)}")
            iter_id += 1
            # NOTE(Soo): If it's not in while loop, it will cause race condition for CP;
            # I couldn't understand the detail though.
            torch.cuda.synchronize()

        torch.distributed.barrier()
        end_time = time.perf_counter()
        if not batch_manager.is_reach_max_batch_size and cfg.p_type != "dp" and rank == 0:
            print(f"It can't reach max batch size ({cfg.max_batch_size})")
        # print(rank, cfg.p_type, batch_manager.n_add_seqs)


        # Calculating elapsed time
        elapsed_time_ms = (end_time - start_time) * 1000
        arr_elapsed_time_ms.append(elapsed_time_ms)
        # if rank == 0:
        # print(f"Elapsed time ({rank}): {elapsed_time_ms} ms")

    arr_elapsed_time_ms = arr_elapsed_time_ms[cfg.n_warmup_iters:]
    avg_elapsed_time_ms = np.mean(arr_elapsed_time_ms)
    if rank == 0:
        print(f"[{cfg.p_type}] Avg Elapsed time ({rank}): {avg_elapsed_time_ms:.3f} ms")
    # print("Time array: ", arr_elapsed_time_ms)
    # print("# tokens : ", n_total_tokens)

    return avg_elapsed_time_ms, n_total_tokens

def save_result_to_file(
        cfg: EvaluationConfig,
        measurement: Tuple[float, float],
):
    eval_logger = EvaluationLogger(cfg.output_file_path)

    # HACK(Soo): Save # of decode iterations as a seq len with following condition
    tmp_seq_len = cfg.max_kv_cache_context_len
    if cfg.n_max_decode_iters > 1000:
        tmp_seq_len = cfg.n_max_decode_iters

    # Add a evaluation data point
    eval_logger.add_result(
        model=cfg.model_name,
        batch_size=cfg.num_seqs,
        seq_len=tmp_seq_len,
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
    avg_latency_ms, n_total_tokens = run_local_model(rank, eval_cfg)
    # Throughput (Generated tokens per second)
    n_tokens_per_sec = n_total_tokens  / (avg_latency_ms / 1000.0)
    # # Latency (per generated token)
    # avg_latency_ms = avg_latency_ms / n_total_tokens

    # Log results
    # Use init_cfg instead of eval_cfg since eval_cfg will change
    # depending on parallelism type
    # NOTE(Soo): Save only the latnecy of critical path among all ranks inside this function.
    result = torch.tensor([avg_latency_ms, n_tokens_per_sec], device=torch.device(f"cuda:{rank}"))
    result_list = [torch.empty_like(result) for _ in range(eval_cfg.n_gpus)]
    torch.distributed.all_gather(result_list, result)
    if rank == 0:
        avg_latency_ms, n_tokens_per_sec = -1, -1
        for result in result_list:
            if avg_latency_ms < result[0]:
                avg_latency_ms = result[0].cpu().item()
                n_tokens_per_sec = result[1].cpu().item()
        save_result_to_file(init_cfg, (avg_latency_ms, n_tokens_per_sec))

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
        n_eval_iters = 3,
        n_warmup_iters = 1,
        output_file_path = "/home/byungsoj/eval_results/result.json",
        rand_seed = 0,
        p_type="cp",

        # Model configs
        dtype = torch.bfloat16,
        model_name = "Llama-7B",
        num_seqs = 1024, # num_seqs * max_kv_cache_context_lens should be less than 1280000

        max_kv_cache_context_len = 1000,
        n_min_decode_iters = 900,
        n_max_decode_iters = 1000,
        max_batch_size = float('inf'),

        # num_layers = 32,
        num_layers = 1,
        llama_cfg = LlamaConfig(), # Load Llama-7B configurations

        # vLLM configs
        # FIXME(Soo): Automatically decide # blocks based on available CUDA memory
        # Note(Soo): 90000 is max # of blocks (1 attn in Catalyst cluster, 16 seqs); should be 80000 with bs of 1024
        # Note(Soo): 1000 is max # of blocks (Llama-7B 32 layers)
        # num_blocks = 1000,
        num_blocks = 80000,
        # num_blocks = 100, # Debug
        block_size = 16,
        partition_size = 512,
    )
    assert eval_cfg.n_max_decode_iters == 1000, "If not, fix Hack in save_result_to_file function"

    # HACK(Soo): There is no limit on context len now because of hack to share KV cache when it overflows
    # Long prefix: Throughput vs. seqnuence length
    # eval_cfg.output_file_path = "/home/byungsoj/eval_results/long-prefix-0225.json"
    # eval_cfg.n_eval_iters = 1  # This is enough to remove variance since # of iterations is large
    # p_types = ["cp", "tp", "dp"]
    # p_types = ["cp"]
    # p_types = ["dp"]
    # p_types = ["tp"]

    # max_kv_cache_context_lens = []
    # max_kv_cache_context_lens.append([i for i in range(50000, 100001, 10000)])
    # max_kv_cache_context_lens.append([i for i in range(10000, 50001, 10000)])
    # max_kv_cache_context_lens.append([i for i in range(10000, 30001, 5000)])
    # num_seqs_arr = [32, 128, 512]
    # max_batch_size_arr = [32, 128, 512]

    # Debug
    # max_kv_cache_context_lens = [[25000, 30000]]#, [10000]]
    # num_seqs_arr = [512]#, 128]
    # max_batch_size_arr = [512]

    # Setting for throughput vs. latency
    # seq_len (10000) - max_batch_size (512), 40000 - 128, 80000 - 64
    # max_kv_cache_context_lens = [10000, 40000, 80000]
    # num_seqs_arr = [16, 32, 64, 128, 256, 512]
    # max_batch_size_arr = [16, 32, 64, 128, 256, 512]

    # for n_idx, (n_seqs, max_bs) in enumerate(zip(num_seqs_arr, max_batch_size_arr)):
    #     for cache_len in max_kv_cache_context_lens[n_idx]:
    #         for p_type in p_types:
    #             eval_cfg.p_type = p_type
    #             eval_cfg.max_kv_cache_context_len = cache_len
    #             eval_cfg.num_seqs = n_seqs
    #             eval_cfg.max_batch_size = max_bs
    #
    #             check_eval_configs(eval_cfg)
    #             torch.multiprocessing.spawn(run_distributed_model,
    #                                         args=(eval_cfg,),
    #                                         nprocs=eval_cfg.n_gpus,
    #                                         join=True)

    # Long decode: Throughput vs. seqnuence length
    eval_cfg.output_file_path = "/home/byungsoj/eval_results/long-decode-0225.json"
    eval_cfg.n_eval_iters = 1 # This is enough to remove variance since # of iterations is large
    eval_cfg.max_kv_cache_context_len = 10000

    p_types = ["cp", "tp", "dp"]
    # p_types = ["cp"]
    # p_types = ["dp"]
    # p_types = ["tp"]

    # n_min_decode_iters_arr, n_max_decode_iters_arr = [], []
    # n_min_decode_iters_arr.append([i for i in range(5000, 45001, 10000)])
    # n_max_decode_iters_arr.append([i for i in range(10000, 50001, 10000)])
    # n_min_decode_iters_arr.append([i for i in range(3000, 23001, 5000)])
    # n_max_decode_iters_arr.append([i for i in range(5000, 25001, 5000)])
    # n_min_decode_iters_arr.append([i for i in range(4500, 9501, 1000)])
    # n_max_decode_iters_arr.append([i for i in range(5000, 10001, 1000)])
    # num_seqs_arr = [32, 128, 512]
    # max_batch_size_arr = [32, 128, 512]

    # Debug
    n_min_decode_iters_arr = [[3500]]
    n_max_decode_iters_arr = [[4000]]
    num_seqs_arr = [512]
    max_batch_size_arr = [512]

    for n_idx, (n_seqs, max_bs) in enumerate(zip(num_seqs_arr, max_batch_size_arr)):
        for n_min_decode_iters, n_max_decode_iters in zip(n_min_decode_iters_arr[n_idx], n_max_decode_iters_arr[n_idx]):
            for p_type in p_types:
                eval_cfg.p_type = p_type
                eval_cfg.n_min_decode_iters = n_min_decode_iters
                eval_cfg.n_max_decode_iters = n_max_decode_iters
                eval_cfg.num_seqs = n_seqs
                eval_cfg.max_batch_size = max_bs

                check_eval_configs(eval_cfg)
                torch.multiprocessing.spawn(run_distributed_model,
                                            args=(eval_cfg,),
                                            nprocs=eval_cfg.n_gpus,
                                            join=True)
