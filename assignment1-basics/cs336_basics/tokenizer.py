import regex as re
import json
import base64
from collections.abc import Iterable, Iterator

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # 复制 vocab
        self.vocab = dict(vocab)
        self.vocab_size = len(self.vocab)

        # 复制 merges
        self.merges = list(merges)

        # 处理special tokens - 长度降序排列
        self.special_tokens = []
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key = len, reverse = True)
            for tok in self.special_tokens:
                tok_bytes = tok.encode("utf-8")
                if tok_bytes not in self.vocab.values():
                    self.vocab[self.vocab_size] = tok_bytes
                    self.vocab_size += 1

        # 实现分割 pattern
        self._special_pattern = None
        if self.special_tokens:
            pattern = "|".join(re.escape(t) for t in self.special_tokens)
            self._special_pattern = re.compile(f"({pattern})")

        # encode bytes -> id
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}

        # 加速 merges 查找
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    # 从文件中获取vocab和merge
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None      
    ) -> "Tokenizer":
        # 读取vocab
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        vocab = {int(k): base64.b64decode(v) for k, v in raw.items()}
        
        # 读取megre
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        merges = [(base64.b64decode(t1), base64.b64decode(t2)) for t1, t2 in raw]
        
        # 调用构造函数
        return cls(vocab, merges, special_tokens)

    # 辅助函数-编码
    def _encode(
            self,
            text: str
    ) -> list[int]:
        # 分割
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        pre_tokens = re.findall(PAT, text)

        # 合并
        tokens = []
        for pre_token in pre_tokens:
            # str -> bytes
            token_bytes = [bytes([b]) for b in pre_token.encode("utf-8")]
            # BPE 合并
            token_bytes = self._bpe_merge(token_bytes)
            # bytes -> token IDs
            for token in token_bytes:
                tokens.append(self.bytes_to_id[token])
        return tokens
    
    # 辅助函数-合并单个pre_token
    def _bpe_merge(
        self,
        seq: list[bytes]
    ) -> list[bytes]:
        while len(seq) >= 2:
            # 寻找rank最小pair
            best_rank = float("inf")
            best_pair = None
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = (i, pair)
            
            if best_pair is None or best_rank == float("inf"):
                break

            # 合并bset pair
            i, _ = best_pair
            merged = seq[i] + seq[i+1]
            seq = seq[:i] + [merged] + seq[i+2:]
        return seq

    # 编码内存文本
    def encode(
        self,
        text: str
    ) -> list[int]:
        if not text:
            return []
        
        if self.special_tokens:
            parts = self._special_pattern.split(text)
        else:
            parts = [text]            

        token_ids = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                token_ids.append(self.bytes_to_id[part.encode("utf-8")])
            else:
                token_ids.extend(self._encode(part))
        return token_ids

    # 分块编码文件    
    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterator[int]:
        for chunk in iterable:
            if not chunk:
                continue
        
            if self.special_tokens:
                # special tokens 切分
                parts = self._special_pattern.split(chunk)

                for part in parts:
                    if not part:
                        continue
                    # special token直接输出
                    if part in self.special_tokens:
                        yield self.bytes_to_id[part.encode("utf-8")]
                    else:
                        for tid in self.encode(part):
                            yield tid
            else:
                for tid in self.encode(chunk):
                    yield tid
    
    # 解码token IDs
    def decode(
        self,
        ids: list[int]
    ) -> str:
        # 词汇表 vocab id -> bytes
        text_bytes = b"".join(self.vocab[i] for i in ids)
        # 处理无效 UTF-8
        return text_bytes.decode("utf-8", errors = "replace")