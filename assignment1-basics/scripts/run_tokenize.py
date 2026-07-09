import os
import sys
import argparse
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.tokenizer import Tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize text data")

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True, help="vocab.txt 路径")
    parser.add_argument("--merges_path", type=str, required=True, help="merges.txt 路径")
    parser.add_argument("--special_tokens", type=str, nargs="+", default=["<|endoftext|>"])
    parser.add_argument("--buffer_size", type=int, default=100_000, help="写npy缓存")

    return parser.parse_args()

# 流式编码文件
def encode_file_streaming(
    tokenizer: Tokenizer, 
    input_path: str,
):
    with open(input_path, "r", encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            yield token_id # yield 即用即算 惰性求值

def main():
    args = parse_args()

    # 加载 tokenizer
    print(f"Loading tokenizer from {args.vocab_path}")
    tokenizer = Tokenizer.from_files(
        vocab_filepath = args.vocab_path,
        merges_filepath = args.merges_path,
        special_tokens = args.special_tokens,
    )
    print(f"Vocab size: {tokenizer.vocab_size}")

    # 流式编码 流式写入
    print(f"\nEncoding {args.input_path}")
    
    buffer = []
    total_count = 0
    start_time = time.time()

    output_file = open(args.output_path, "wb")

    for token_id in encode_file_streaming(tokenizer, args.input_path):
        buffer.append(token_id)
        total_count += 1

        # buffer 存满 写入
        if len(buffer) >= args.buffer_size:
            batch_np = np.array(buffer, dtype=np.int32)
            batch_np.tofile(output_file)
            buffer = []
        
        if total_count % 1_000_000 == 0:
            elapsed = time.time() - start_time
            print(f"Encoded {total_count:,} tokens in {elapsed:.1f}s")
    
    # 最后一批缓存数据
    if buffer:
        batch_np = np.array(buffer, dtype=np.int32)
        batch_np.tofile(output_file)
    
    output_file.close()

    elapsed = time.time() - start_time
    print(f"\nTotal {total_count:,} tokens in {elapsed:.1f}s")
    print(f"Saved to {args.output_path}")

if __name__ == "__main__":
    main()