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
            train.append(items)
        else:
            train.append( items[:-2] )
            valid.append( (items[:-2], items[-2]) )
            test.append( (items[:-1], items[-1]) )
    
    return [train, valid, test, num_users, num_items]


def PadOrTruncate(max_len: int, pad_value: int = 0):
    
    def _inner(x: List[int]):
        if len(x) > max_len:
            return x[:max_len]
        
        return [pad_value] * (max_len - len(x)) + x
    
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


def batchify(data: List, batch_size: int, max_seq_len: int, num_items: int, seed: int = 42):
    random.seed(seed)
    random.shuffle(data)
    pad_or_truncate = PadOrTruncate(max_len=max_seq_len)
    
    for batch in iterutils.chunked(data, size=batch_size):
        seq = [pad_or_truncate(e[:-1]) for e in batch]
        pos = [pad_or_truncate(e[1:]) for e in batch]
        negs = [pad_or_truncate(negative_sampling(e, num_items)) for e in batch]
        yield (seq, pos, negs)


def batchify_test(data: List, batch_size: int, max_seq_len: int, num_items: int, 
                  num_neg_labels: int = 100):
    pad_or_truncate = PadOrTruncate(max_len=max_seq_len)
    
    for batch in iterutils.chunked(data, size=batch_size):
        seq = [pad_or_truncate(e) for e, _ in batch]
        labels = [[pos_label] + [randint(1, num_items+1, e) for i in range(num_neg_labels)] 
                  for e, pos_label in batch]
        yield (seq, labels)


