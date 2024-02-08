import json
import torch

from typing import Optional
from transformers import LlamaConfig

class EvaluationConfig:
    def __init__(
        self,
        # Evaluation configs
        n_gpus: int,
        n_eval_iters: int,
        n_warmup_iters: int,
        output_file_path: str,
        rand_seed: int,
        p_type: str,

        # Model configs
        dtype: torch.dtype,
        model_name: str,
        num_seqs: int,
        max_kv_cache_context_len: int,
        num_layers: int,
        llama_cfg: LlamaConfig,

        # vLLM configs
        num_blocks: int,
        block_size: int,
        partition_size: int,
    ) -> None:
        self.n_gpus = n_gpus
        self.n_eval_iters = n_eval_iters
        self.n_warmup_iters = n_warmup_iters
        self.output_file_path = output_file_path
        self.rand_seed = rand_seed
        self.p_type = p_type

        self.dtype = dtype
        self.model_name = model_name
        self.num_seqs = num_seqs
        self.max_kv_cache_context_len = max_kv_cache_context_len
        self.num_layers = num_layers
        self.llama_cfg = llama_cfg

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.partition_size = partition_size

class EvaluationLogger:
    def __init__(self, filename):
        self.filename = filename
        self.results = {}

        # Read results if JSON file already exists
        try:
            with open(self.filename, 'r') as file:
                self.results = json.load(file)
        except FileNotFoundError:
            print(f"Create {self.filename} since it does not exist.")
        except json.JSONDecodeError:
            print(f"The file {self.filename} is not valid JSON.")

    def add_result(self, model, batch_size, seq_len, p_type, measurement):
        # To match the data type of keys read from JSON file
        # This prevents adding two same key to a dictionary
        batch_size, seq_len = str(batch_size), str(seq_len)
        avg_latency_ms, n_tokens_per_sec = measurement

        if model not in self.results:
            self.results[model] = {}

        if batch_size not in self.results[model]:
            self.results[model][batch_size] = {}

        if seq_len not in self.results[model][batch_size]:
            self.results[model][batch_size][seq_len] = {}

        self.results[model][batch_size][seq_len][p_type] = (round(avg_latency_ms, 3), int(n_tokens_per_sec))
        self.results[model][batch_size][seq_len] = dict(sorted(self.results[model][batch_size][seq_len].items()))

    def save_results(self):
        with open(self.filename, 'w') as file:
            json.dump(self.results, file, indent=2)
        print(f"Results saved to {self.filename}")