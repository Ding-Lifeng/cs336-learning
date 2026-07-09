import os
import sys
import argparse
import time
import json
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加 python 模块搜索路径

from cs336_basics.bpe import train_bpe

# 命令解析
def parse_args():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")

    parser.add_argument("--input_path", type=str, required=True, help="训练文本路径")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--special_tokens", type=str, nargs="+", default=["<|endoftext|>"], help="特殊 token 列表")
    parser.add_argument("--vocab_output", type=str, default="output/vocab.json", help="vocab 输出路径")
    parser.add_argument("--merges_output", type=str, default="output/merges.json", help="merges 输出路径") 

    return parser.parse_args()   

# 文件形式保存 vocab
def save_vocab(
    vocab: dict[int, bytes],
    path:str,
    ):
    # 每行一个 token
    # 排序依据 id
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    vocab_b64 = {tid: base64.b64encode(tok).decode("ascii") for tid, tok in sorted(vocab.items())}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab_b64, f, ensure_ascii=False)

# 文件形式保存 merges
def save_merges(
    merges: list[tuple[bytes, bytes]],
    path: str,
):
    # token1 token2
    merges_b64 = [(base64.b64encode(t1).decode("ascii"),
                   base64.b64encode(t2).decode("ascii"))
                   for t1, t2 in merges]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merges_b64, f, ensure_ascii=False)

def main():
    args = parse_args()
    
    # 训练 BPE
    print(f"Training BPE tokenizer on {args.input_path}...")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Special tokens: {args.special_tokens}")

    start_time = time.time()
    vocab, merges = train_bpe(
        input_path = args.input_path,
        vocab_size = args.vocab_size,
        special_tokens = args.special_tokens,
    )
    elapsed = time.time() - start_time # BPE 训练时间

    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Final vocab size: {len(vocab)}")

    # 存储 vocab 和 merges
    print(f"\n Saving to {args.vocab_output} and {args.merges_output}")
    save_vocab(vocab, args.vocab_output)
    save_merges(merges, args.merges_output)
    print(f"Saved vocab ({len(vocab)} entries)")
    print(f"Saved merges ({len(merges)} pairs)")

if __name__ == "__main__":
    main()