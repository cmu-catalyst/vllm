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

a = Sequence(0, 1, 2)
b = Sequence(1, 2, 3)

from sortedcontainers import SortedList
import copy

running_queue = []


running_queue.append(a)
running_queue.append(b)
print(running_queue.pop(0))
tmp_r_q = copy.deepcopy(running_queue)
print(len(running_queue))
running_queue.remove(Sequence(1, 4, 3))
print(len(running_queue))
