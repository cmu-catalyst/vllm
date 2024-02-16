import copy
import torch

from sortedcontainers import SortedList
from synthetic_data_utils import gen_block_table_and_slot_mapping
from vllm.model_executor.input_metadata import InputMetadata

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

class KVCacheManager:
    def __init__(self, cfg, hidden_size, device, kv_caches):
        # HACK(Soo): Assume KV cache for single layer
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.device = device
        self.value_cache = kv_caches[1] # (# blocks, # heads, head size, block size)

        self.avail_gpu_cache_space = cfg.num_blocks * cfg.block_size # # of tokens to store
        if cfg.p_type == "tp":
            self.avail_gpu_cache_space *= cfg.n_gpus

    def add_seq(self, seq):
        if self.avail_gpu_cache_space >= seq.cur_len:
            self.avail_gpu_cache_space -= seq.cur_len

            # HACK(Soo): Only to measure CPU transfer time
            tmp_data_cpu = torch.empty(2, seq.cur_len, self.hidden_size, dtype=self.cfg.dtype, device='cpu')
            tmp_data_cpu.uniform_(-1, 1)
            tmp_data_gpu = tmp_data_cpu.to(self.device)
            tmp_data_gpu = tmp_data_gpu + 1  # Dummy compute to prevent lazy transfer
        else:
            return False

        return True

    def evict_seq_to_cpu(self, seq):
        self.avail_gpu_cache_space += seq.cur_len

        # HACK(Soo): Only to measure CPU transfer time
        tmp_data_gpu = torch.empty(2, seq.cur_len, self.hidden_size, dtype=self.cfg.dtype, device=self.device)
        tmp_data_gpu.uniform_(-1, 1)
        tmp_data_cpu = tmp_data_gpu.cpu()
        tmp_data_cpu = tmp_data_cpu + 1 # Dummy compute to prevent lazy transfer

    def discard_seq(self, seq):
        self.avail_gpu_cache_space += seq.cur_len

    def add_new_token(self):
        if self.avail_gpu_cache_space > 0:
            self.avail_gpu_cache_space -= 1
        else:
            return False

        return True



class BatchManager:
    def __init__(self, cfg, rank, device, hidden_size, seq_lens, target_seq_lens, kv_caches):
        self.cfg = cfg
        self.rank = rank
        self.device = device
        self.hidden_size = hidden_size

        self.running_queue = SortedList(key = lambda seq: seq.cur_len)
        self.wait_queue = SortedList(key = lambda seq: -seq.cur_len)

        # Initialize KV cache manager
        # HACK(Soo): We secure block for target_seq_len in advance to make implementation easier
        self.imaginary_all_block_tables, _ = gen_block_table_and_slot_mapping(
            cfg.num_blocks, len(target_seq_lens), target_seq_lens, cfg.block_size, device)
        self.kv_cache_manager = KVCacheManager(cfg, hidden_size, device, kv_caches)
        self.target_gen_iters = []

        # Initialize wait queue
        for idx, (sl, tsl) in enumerate(zip(seq_lens, target_seq_lens)):
            self.wait_queue.add(Sequence(idx, sl, tsl))
            self.target_gen_iters.append(tsl - sl.cpu().item())

        self.load_seqs_to_gpu()

    def is_running(self):
        return self.running_queue or self.wait_queue

    def discard_completed_seqs(self, iter_id):
        tmp_running_queue = copy.deepcopy(self.running_queue)
        for seq in tmp_running_queue:
            if self.cfg.p_type == "cp":
                # HACK(Soo): Discard based on iter_id with assumption that generation length is always multiple of n_gpus
                if self.target_gen_iters[seq.idx] == iter_id:
                    self.running_queue.remove(seq)
                    self.kv_cache_manager.discard_seq(seq)
            else:
                if seq.cur_len >= seq.target_len:
                    self.running_queue.remove(seq)
                    self.kv_cache_manager.discard_seq(seq)

    def evict_seq_to_cpu_and_put_it_on_hold(self):
        seq = self.running_queue.pop(0)
        self.kv_cache_manager.evict_seq_to_cpu(seq)
        self.wait_queue.add(seq)


    def add_new_tokens(self, iter_id):
        for seq in self.running_queue:
            cp_special_cond = self.cfg.p_type == "cp" and self.rank == (iter_id % self.cfg.n_gpus)
            if self.cfg.p_type != "cp" or cp_special_cond:
                seq.cur_len += 1
                while not self.kv_cache_manager.add_new_token():
                    self.evict_seq_to_cpu_and_put_it_on_hold()
    def load_seqs_to_gpu(self):
        tmp_wait_queue = copy.deepcopy(self.wait_queue)
        for seq in tmp_wait_queue:
            if self.kv_cache_manager.add_seq(seq):
                self.running_queue.add(seq)
                self.wait_queue.remove(seq)

    def update(self, iter_id):
        # Operations from current model execution
        self.discard_completed_seqs(iter_id)
        self.add_new_tokens(iter_id)
        self.load_seqs_to_gpu()

    def gen_hidden_states(self, scale):
        decode_context_len = 1

        kv_cache_num_seqs = len(self.running_queue)
        input_num_seqs = kv_cache_num_seqs // self.cfg.n_gpus if self.cfg.p_type == "cp" else kv_cache_num_seqs
        if self.cfg.p_type == "cp" and self.rank < (kv_cache_num_seqs % self.cfg.n_gpus):
            input_num_seqs += 1

        hidden_states = torch.empty(input_num_seqs, decode_context_len, self.hidden_size,
                                    dtype=self.cfg.dtype, device=self.device)
        hidden_states.uniform_(-scale, scale)

        return hidden_states

    def gen_input_metadata(self):
        cur_block_tables = []
        tmp_running_queue = list(copy.deepcopy(self.running_queue))
        context_lens = []
        max_context_len = 0
        for seq in tmp_running_queue:
            cur_block_tables.append(self.imaginary_all_block_tables[seq.idx])
            context_lens.append(seq.cur_len)
            if max_context_len < seq.cur_len:
                max_context_len = seq.cur_len

        context_lens = torch.tensor(context_lens, dtype=torch.int, device=self.device)
        cur_block_tables = torch.tensor(cur_block_tables, dtype=torch.int, device=self.device)

        input_metadata = InputMetadata(
            is_prompt=False,
            slot_mapping=None,
            max_context_len=max_context_len,
            context_lens=context_lens,
            block_tables=cur_block_tables,
            use_cuda_graph=False,
        )

        return input_metadata
