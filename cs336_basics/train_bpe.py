import os
import regex as re

# pre-tokenize
def split_with_special_tokens(text, special_tokens) -> list[str]:
    sorted_specials = sorted(special_tokens, key = len, reverse = True)
    escaped = [re.escape(tok) for tok in sorted_specials] 
    pattern = "|".join(escaped)

    parts = re.split(f"({pattern})", text)
    
    return parts

# pre-tokenize
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

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # 全文读取
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 根据特殊token切分
    parts = split_with_special_tokens(text, special_tokens)

    # pre-tokenize
    pretokens = get_pretokens(parts,  special_tokens)

    # pretoken频率统计
    special_set = set(special_tokens)
    pretoken_freq: dict[str, int] = {}
    pretoken_content: dict[str, list[bytes]] = {}
    
    for pretoken in pretokens:
        if pretoken not in pretoken_freq:
            if pretoken in special_set:
                pretoken_content[pretoken] = [pretoken.encode("utf-8")]
            else:
                pretoken_content[pretoken] = [bytes([b]) for b in pretoken.encode("utf-8")]
        pretoken_freq[pretoken] = pretoken_freq.get(pretoken, 0) + 1

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