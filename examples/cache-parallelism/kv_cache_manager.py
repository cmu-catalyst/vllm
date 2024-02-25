import copy
import torch
from typing import Any, Dict, List, Tuple

from synthetic_data_utils import gen_block_table_and_slot_mapping

from vllm._C import cache_ops
from vllm.model_executor.input_metadata import InputMetadata
from llama_llm import KVCache

class Sequence:
    def __init__(self, idx, cur_len, target_len):
        self.idx = idx
        self.cur_len = cur_len
        self.target_len = target_len

    def __lt__(self, other):
        return self.cur_len < other.cur_len

    def __eq__(self, other):
        return self.idx == other.idx

    def __hash__(self):
        return hash(self.idx)

    def __repr__(self):
        return str(self.idx)

class KVCacheManager:
    def __init__(self, cfg, hidden_size, device, kv_caches, cpu_kv_caches):
        # HACK(Soo): Assume KV cache for single layer
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.hidden_size_per_gpu = hidden_size // cfg.n_gpus if cfg.p_type == "tp" else hidden_size
        self.device = device

        # For cache swap btw CPU and GPU
        self.gpu_key_cache = kv_caches[0] # (# blocks, # heads, head size // x, block size, x)
        self.gpu_value_cache = kv_caches[1] # (# blocks, # heads, head size, block size)
        self.cpu_key_cache = cpu_kv_caches[0]
        self.cpu_value_cache = cpu_kv_caches[1]

        self.avail_gpu_cache_space = cfg.num_blocks * cfg.block_size # # of tokens to store
        if cfg.p_type == "tp":
            self.avail_gpu_cache_space *= cfg.n_gpus
        self.ref_total_gpu_cache_space = self.avail_gpu_cache_space


    def swap_cpu_to_gpu(self, total_seq_len):
        # HACK(Soo): Only to measure CPU transfer time
        assert total_seq_len <= self.ref_total_gpu_cache_space
        n_blocks = (total_seq_len + self.cfg.block_size - 1) // self.cfg.block_size
        # Dummy compute to prevent no copy
        self.gpu_key_cache[:n_blocks,:,:,:,:] = self.cpu_key_cache[:n_blocks,:,:,:,:] + 0.001
        self.gpu_value_cache[:n_blocks, :, :, :] = self.cpu_value_cache[:n_blocks, :, :, :] + 0.001

    def add_seq(self, seq):
        if self.avail_gpu_cache_space >= seq.cur_len:
            self.avail_gpu_cache_space -= seq.cur_len
        else:
            return False

        return True

    def evict_seq_to_cpu(self, seq):
        self.avail_gpu_cache_space += seq.cur_len

        # print("[Eviction] Please do sanicy check for eviction once for long generation regime!")
        # HACK(Soo): Only to measure CPU transfer time
        n_blocks = (seq.cur_len + self.cfg.block_size - 1) // self.cfg.block_size
        # Dummy compute to prevent no copy
        self.cpu_key_cache[:n_blocks, :, :, :, :] = self.gpu_key_cache[:n_blocks, :, :, :, :] + 0.001
        self.cpu_value_cache[:n_blocks, :, :, :] = self.gpu_value_cache[:n_blocks, :, :, :] + 0.001

    def discard_seq(self, seq):
        self.avail_gpu_cache_space += seq.cur_len

    def add_new_token(self):
        if self.avail_gpu_cache_space > 0:
            self.avail_gpu_cache_space -= 1
        else:
            return False

        return True



class BatchManager:
    def __init__(self, cfg, rank, device, hidden_size, seq_lens, target_seq_lens, kv_caches, cpu_kv_caches):
        self.cfg = cfg
        self.rank = rank
        self.device = device
        self.hidden_size = hidden_size
        self.max_batch_size = cfg.max_batch_size

        # WARNING(SOO): Do not use SortedList! It does not use __eq__ in Sequence and cause errors!
        self.running_queue = []
        self.wait_queue = []

        # Initialize KV cache manager
        # HACK(Soo): We secure block for target_seq_len in advance to make implementation easier
        self.imaginary_all_block_tables, _ = gen_block_table_and_slot_mapping(
            cfg.num_blocks, len(target_seq_lens), target_seq_lens, cfg.block_size, device)
        self.kv_cache_manager = KVCacheManager(cfg, hidden_size, device, kv_caches, cpu_kv_caches)
        self.target_gen_iters = []

        # Initialize wait queue
        # FIXME(Soo): I don't understand why seq_lens should be torch.tensor to remove ValueError for self.running_queue(seq)
        for idx, (sl, tsl) in enumerate(zip(seq_lens, target_seq_lens)):
            self.wait_queue.append(Sequence(idx, sl.cpu().item(), tsl))
            self.target_gen_iters.append(tsl - sl.cpu().item())

        # print("Target gen iters: ", self.target_gen_iters)
        # DEBUG
        self.n_add_seqs = 0
        self.load_seqs_to_gpu()

    def is_running(self):
        return self.running_queue or self.wait_queue

    def discard_completed_seqs(self, iter_id):
        tmp_running_queue = copy.deepcopy(self.running_queue)
        for seq in tmp_running_queue:
            # if self.cfg.p_type == "cp":
            #     # HACK(Soo): Discard based on iter_id with assumption that generation length is always multiple of n_gpus
            #     if self.target_gen_iters[seq.idx] * self.cfg.n_gpus == iter_id:
            #         self.running_queue.remove(seq)
            #         self.kv_cache_manager.discard_seq(seq)
            # else:

            # HACK(Soo): This code is simpler, but gives CP (n_gpus - 1) less iterations / seq (it's minor though)
            if seq.cur_len + 1 >= seq.target_len:
                self.running_queue.remove(seq)
                self.kv_cache_manager.discard_seq(seq)

    def evict_seq_to_cpu_and_put_it_on_hold(self):
        seq = self.running_queue.pop(0)
        # print("Seq ID to evict: ", seq.idx, "len: ", seq.cur_len, seq.target_len)
        self.kv_cache_manager.evict_seq_to_cpu(seq)
        self.wait_queue.append(seq)


    def add_new_tokens(self, iter_id):
        for seq in self.running_queue:
            # HACK(Soo): Prevent discrepancy in sequence length across GPUs that causes illegal memory access
            cp_special_cond = iter_id % self.cfg.n_gpus == 0
            if self.cfg.p_type != "cp" or cp_special_cond:
                seq.cur_len += 1
                while not self.kv_cache_manager.add_new_token():
                    self.evict_seq_to_cpu_and_put_it_on_hold()

    def load_seqs_to_gpu(self):
        assert self.max_batch_size >= len(self.running_queue)
        if self.max_batch_size == len(self.running_queue):
            return

        tmp_wait_queue = copy.deepcopy(self.wait_queue)
        tmp_total_seq_len = 0
        for seq in tmp_wait_queue:
            if self.kv_cache_manager.add_seq(seq):
                self.n_add_seqs += 1
                tmp_total_seq_len += seq.cur_len
                self.running_queue.append(seq)
                self.wait_queue.remove(seq)
                if self.max_batch_size == len(self.running_queue):
                    return

        # TODO(Soo): Replace it with real KV cache swap logic
        if tmp_total_seq_len > 0:
            self.kv_cache_manager.swap_cpu_to_gpu(tmp_total_seq_len)

        assert self.running_queue or not self.wait_queue, "OOM"

    def update(self, iter_id):
        # Operations from current model execution
        self.discard_completed_seqs(iter_id)
        self.add_new_tokens(iter_id)
        self.load_seqs_to_gpu()
        # print(f"Iter id {iter_id}: ", len(self.running_queue), len(self.wait_queue))

    def get_input_num_seqs(self):
        # HACK(Soo): Create redundant input to prevent illegal memory access from all_gather for CP case
        kv_cache_num_seqs = len(self.running_queue)
        input_num_seqs = (kv_cache_num_seqs + self.cfg.n_gpus - 1) // self.cfg.n_gpus if self.cfg.p_type == "cp" else kv_cache_num_seqs

        return input_num_seqs

    def gen_hidden_states(self, scale):
        decode_context_len = 1

        input_num_seqs = self.get_input_num_seqs()
        hidden_states = torch.empty(input_num_seqs, decode_context_len, self.hidden_size,
                                    dtype=self.cfg.dtype, device=self.device)
        hidden_states.uniform_(-scale, scale)

        # TODO(Soo): Make redundant input to zero vector for CP case?
        # print(f"hidden shape (rank {self.rank}): ", hidden_states.shape)

        return hidden_states

    def gen_input_metadata(self):
        cur_block_tables = []
        context_lens = []
        max_context_len = 0

        input_num_seqs = self.get_input_num_seqs()
        kv_cache_num_seqs = len(self.running_queue)
        for seq in self.running_queue:
            cur_block_tables.append(self.imaginary_all_block_tables[seq.idx])
            context_lens.append(seq.cur_len)
            if max_context_len < seq.cur_len:
                max_context_len = seq.cur_len

        # HACK(Soo): Create redundant input to prevent illegal memory access from all_gather for CP case
        while self.cfg.p_type == "cp" and kv_cache_num_seqs < input_num_seqs * self.cfg.n_gpus:
            context_lens.append(1)
            cur_block_tables.append(self.imaginary_all_block_tables[0])
            kv_cache_num_seqs += 1

        # print(f"conlen shape (rank {self.rank}): ", len(context_lens))
        # print(f"bt shape (rank {self.rank}): ", len(cur_block_tables))
        context_lens = torch.tensor(context_lens, dtype=torch.int, device=self.device)
        cur_block_tables = torch.tensor(cur_block_tables, dtype=torch.int, device=self.device)

        # print(f"conlen shape (rank {self.rank}): ", context_lens.shape)
        # print(f"bt shape (rank {self.rank}): ", cur_block_tables.shape)

        input_metadata = InputMetadata(
            is_prompt=False,
            slot_mapping=None,
            max_context_len=max_context_len,
            context_lens=context_lens,
            block_tables=cur_block_tables,
            use_cuda_graph=False,
        )

        return input_metadata
