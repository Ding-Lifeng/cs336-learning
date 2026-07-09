import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool
from collections import Counter

# 文件分块
def find_chunk_boundaries(
    file: BinaryIO,
    chunk_size: int,
    split_special_token: bytes,
) -> list[int]:
    file.seek(0, os.SEEK_END) # 移动文件指针到末尾
    file_size = file.tell()
    file.seek(0)

    # num_chunks = max(1, -(-file_size // chunk_size)) # 双负号向上取整
    num_chunks = max(1, file_size // chunk_size)
    chunk_boundaries = [i * chunk_size for i in range(num_chunks + 1)]
    chunk_boundaries[-1] = file_size # 末尾对齐 EOF
    
    mini_chunk_size = 4096 # special_token 查找步长

    for i in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[i]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[i] = file_size
                break
            found_at = mini_chunk.find(split_special_token) # mini_chunk 中定位special_token ，开展分块
            if found_at != -1:
                chunk_boundaries[i] = initial_position + found_at # 实际依据 special_token 分块
                break
            initial_position += mini_chunk_size # 继续查找分块边界
    
    return sorted(set(chunk_boundaries)) # 去重 恢复排序

# pre-tokenize
def split_with_special_tokens(text, special_tokens) -> list[str]:
    sorted_specials = sorted(special_tokens, key = len, reverse = True)
    escaped = [re.escape(tok) for tok in sorted_specials] 
    pattern = "|".join(escaped)

    parts = re.split(f"({pattern})", text)
    
    return parts

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""    

def get_pretokens(parts, special_tokens) -> list[str]:    
    pretokens = []
    special_set = set(special_tokens)
    for part in parts:
        if not part:
            continue
        elif part in special_set:
            pretokens.append(part)
        else:
            pretokens.extend(re.findall(PAT, part))
    return pretokens

# 合并pretoken中的byte pair
def merge_pair_in_seq(
        seq: list[bytes],
        max_pairs: tuple[bytes, bytes],
) -> list[bytes]:
    new_list = []
    i = 0
    while i < len(seq):
        if i < len(seq)-1 and seq[i] == max_pairs[0] and seq[i+1] == max_pairs[1]:
            new_list.append(max_pairs[0] + max_pairs[1])
            i += 2    
        else:
            new_list.append(seq[i])
            i += 1
    
    return new_list 

# 多进程 - 每个进程读一个文件
def _pretokenize_chunk_worker(args: tuple) -> Counter:
    path, start, end, special_tokens, _worker_id = args
    with open(path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    text = chunk_bytes.decode("utf-8", errors="replace")

    parts = split_with_special_tokens(text, special_tokens)
    pretokens = get_pretokens(parts, special_tokens)
    return Counter(pretokens) # 分块计数

# 多进程流式 pretokenize
def _stream_pretokenize_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    pretoken_freq: dict[str, int],
    pretoken_content: dict[str, list[bytes]],
    num_processes: int = 4,
    chunk_size_bytes: int = 10 * 1024 * 1024 # 10 MB
) -> None:
    # 取切块对齐点
    split_token = special_tokens[0].encode("utf-8")

    # 切块
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, chunk_size_bytes, split_token)

    # 多进程任务
    tasks = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        tasks.append((str(input_path), start, end, special_tokens, i))

    with Pool(processes = num_processes) as pool:
        for local_counter in pool.imap_unordered(_pretokenize_chunk_worker, tasks):
            for pt, cnt in local_counter.items():
                pretoken_freq[pt] = pretoken_freq.get(pt, 0) + cnt # 词频统计

    # 构建 pretoken_content
    special_set = set(special_tokens)
    for pt in pretoken_freq:
        if pt in special_set:
            pretoken_content[pt] = [pt.encode("utf-8")]
        else:
            pretoken_content[pt] = [bytes([b]) for b in pt.encode("utf-8")]

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 4,
    chunk_size_bytes: int = 10 * 1024 * 1024,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 多进程 pretokenize
    pretoken_freq: dict[str, int] = {}
    pretoken_content: dict[str, list[bytes]] = {}

    _stream_pretokenize_parallel(
        input_path = input_path,
        special_tokens = special_tokens,
        pretoken_freq = pretoken_freq,
        pretoken_content = pretoken_content,
        num_processes = num_processes,
        chunk_size_bytes = chunk_size_bytes
    )

    # 初始化vocab(解码)
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # 初始化merges
    merges = []

    # 初始化pair_freq
    pair_freq: dict[tuple[bytes, bytes], int] = {}

    # 初始化反向索引
    pair_to_pretokens: dict[tuple[bytes, bytes], set[str]] = {}

    # 计算pair_freq，建立反向索引
    for pretoken, freq in pretoken_freq.items():
        seq = pretoken_content[pretoken]
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])
            pair_freq[pair] = pair_freq.get(pair, 0) + freq
            pair_to_pretokens.setdefault(pair, set()).add(pretoken)

    # BPE完善vocab
    while len(vocab) < vocab_size:
        if not pair_freq:
            break

        # 查找高频pair (tie-breaking)
        max_pair = max(pair_freq, key = lambda p : (pair_freq[p], p))

        # 更新 vocab 和 merges
        new_token = max_pair[0] + max_pair[1]
        vocab[len(vocab)] = new_token
        merges.append(max_pair)

        # 增量更新
        affected = pair_to_pretokens.pop(max_pair, set()).copy() # 清理反向索引

        # 清理max_pair
        pair_freq.pop(max_pair, None)

        for pretoken in affected:
            seq = pretoken_content[pretoken]
            freq = pretoken_freq[pretoken]

            # 合并pretoken中的max_pair
            new_seq = merge_pair_in_seq(seq, max_pair)
            pretoken_content[pretoken] = new_seq

            # 清除旧seq的pair计数
            for i in range(len(seq) - 1):
                old_pair = (seq[i], seq[i+1])

                if(old_pair == max_pair):
                    continue

                pair_freq[old_pair] -= freq
                if pair_freq[old_pair] <= 0:
                    del pair_freq[old_pair]
                    pair_to_pretokens[old_pair].discard(pretoken)

            # 增加新seq的pair计数
            for i in range(len(new_seq) - 1):
                new_pair = (new_seq[i], new_seq[i+1])
                pair_freq[new_pair] = pair_freq.get(new_pair, 0) + freq
                pair_to_pretokens.setdefault(new_pair, set()).add(pretoken)

    return vocab, merges