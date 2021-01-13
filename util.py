from typing import List
from collections import defaultdict
import numpy as np
from boltons import iterutils
import random


def load_train_val_test(dataset: str) -> List:
    num_users = 0
    num_items = 0
    user_to_items = defaultdict(list)

    # assume user/item index starting from 1
    with open(f"data/{dataset}.txt", "r") as f:
        for line in f:
            u, i = line.rstrip().split(" ")
            u = int(u)
            i = int(i)
            num_users = max(u, num_users)
            num_items = max(i, num_items)
            user_to_items[u].append(i)

    train = []
    valid = []
    test = []
    for user in user_to_items:
        items = user_to_items[user]
        n_items = len(items)
        if n_items <= 1: 
            continue
            
        if n_items < 3:
            train.append( (user, items) )
        else:
            train.append( (user, items[:-2]) )
            valid.append( (user, items[:-2], items[-2]) )
            test.append( (user, items[:-1], items[-1]) )
    
    return [user_to_items, train, valid, test, num_users, num_items]


def PadOrTruncate(max_len: int, pad_value: int = 0, 
                  pad_direction: str = "right", truncate_direction: str = "left"):
    
    def _inner(x: List[int]):
        if len(x) > max_len:
            if truncate_direction == "right":
                return x[:max_len]
            elif truncate_direction == "left":
                return x[-max_len:]
            else:
                raise NotImplementedError(f"Unknown option {truncate_direction}")
        
        if pad_direction == "left":
            return [pad_value] * (max_len - len(x)) + x
        elif pad_direction == "right":
            return x + [pad_value] * (max_len - len(x))
        else:
            raise NotImplementedError(f"Unknown option {pad_direction}")
    
    return _inner


def randint(low: int, high: int, set_to_avoid: List[int]):
    t = np.random.randint(low, high)
    while t in set_to_avoid:
        t = np.random.randint(low, high)
    return t


def negative_sampling(seq: List, num_items: int):
    "Sample n - 1 elements randomly in [1, num_items] but avoid elments in `seq`"
    n = len(seq)
    return [randint(1, num_items+1, seq) for i in range(n-1)]


class DataLoader:
    
    def __init__(self, data: List, batch_size: int, max_seq_len: int, seed: int = 42):
        self.data = data
        self.batch_size = batch_size
        self.pad_or_truncate = PadOrTruncate(max_len=max_seq_len)
        # self.num_items = num_items
        self.seed = seed
        random.seed(seed)
        
    def __iter__(self):
        random.shuffle(self.data)
        for batch in iterutils.chunked(self.data, size=self.batch_size):
            seq = [self.pad_or_truncate(e[:-1]) for _, e in batch]
            pos = [self.pad_or_truncate(e[1:]) for _, e in batch]
            # negs = [self.pad_or_truncate(negative_sampling(e, self.num_items)) for e in batch]
            yield (seq, pos)


def batchify_test(data: List, batch_size: int, max_seq_len: int):
    pad_or_truncate = PadOrTruncate(max_len=max_seq_len)
    
    for batch in iterutils.chunked(data, size=batch_size):
        seq = [pad_or_truncate(e) for _, e, _ in batch]
        pos = [pos_label for _, _, pos_label in batch]
        # pos = [[pos_label + [randint(1, num_items+1, e) for i in range(num_neg_labels)] for e, pos_label in batch]
        yield (seq, pos)


