import argparse
import numpy as np
import torch
import time
import sys
import os

import csv
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, lr_cosine_schedule
from cs336_basics.data import data_loader
from cs336_basics.serialization import save_checkpoint, load_checkpoint
from cs336_basics.nn_utils import cross_entropy, gradient_clipping

# 命令解析
def parse_args():
    parser = argparse.ArgumentParser(description="Tran Transformer LM")

    # 指定参数
    parser.add_argument("--train_data", type=str, required=True, help="训练数据路径")
    parser.add_argument("--valid_data", type=str, default=None, help="验证数据路径")

    # 模型参数
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344) # 接近d_model * 8 / 3 的 64 倍数
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=10000) # 训练次数上限
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--cosine_cycle_iters", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # 评估参数
    parser.add_argument("--eval_interval", type=int, default=500) # 每 n 次迭代评估一次
    parser.add_argument("--eval_iters", type=int, default=50) # 评估采样 50 batch

    # checkpoint 参数
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 恢复")

    # 设备
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42) # 随机数种子

    return parser.parse_args()

# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 模型评估0
def evaluate(model, dataset, args):
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        x, y = data_loader(dataset, args.batch_size, args.context_length, args.device)
        with torch.no_grad(): # 禁用梯度计算
            logits = model(x)
            logits = rearrange(logits, '... vocab -> (...) vocab')
            loss = cross_entropy(logits, y.view(-1))
        losses.append(loss.item())
    model.train() # 恢复训练模式 - 随机丢弃神经元防止过拟合
    return sum(losses) / len(losses)

# 加载数据
def load_data(path):
    return np.memmap(path, dtype=np.int32, mode='r')

# 绘图函数 plot training log
def plot_training_log(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14,5))

    # Loss 曲线图
    ax = axes[0]
    ax.plot(df["iter"], df["train_loss"], label="train loss", alpha=0.5, color="blue")
    val_df = df.dropna(subset=["val_loss"])
    if len(val_df) > 0:
        ax.plot(val_df["iter"], val_df["val_loss"], label="val loss", marker="o", color="red")
    ax.set_xlabel("iter")
    ax.set_ylabel("loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # LR 曲线
    ax = axes[1]
    ax.plot(df["iter"], df["lr"], label="lr", color="green")
    ax.set_xlabel("iter")
    ax.set_ylabel("lr")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot tp {output_path}")

def main():
    args = parse_args()

    # 初始化日志
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["iter", "train_loss", "val_loss", "lr", "elapsed_sec"])

    log_file.flush()
    print(f"Logging to {log_path}")

    # 设置随机种子
    setup_seed(args.seed)

    # 加载数据
    print(f"Loading training data from {args.train_data}...")
    train_data = load_data(args.train_data)
    print(f"Total tokens: {len(train_data):,}") # :,表示数据分隔符

    valid_data = None
    if args.valid_data:
        print(f"Loading validation data from {args.valid_data}...")
        valid_data = load_data(args.valid_data)
        print(f"Total tokens: {len(valid_data):,}")

    # 创建模型
    print("\nCreating model...")
    model = TransformerLM(
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        d_model = args.d_model,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        rope_theta = args.rope_theta
    ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters()) # 模型参数
    print(f"Total parameters: {num_params:,}")

    # 创建优化器
    optimizer = AdamW(
        model.parameters(),
        lr = args.lr,
        eps = 1e-8,
        weight_decay = args.weight_decay,
    )

    # 恢复 checkpoint
    start_iter = 0
    if args.resume and os.path.exists(args.checkpoint_path):
        start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    # Training Loop
    print(f"\nStarting training on {args.device}")
    print(f"Max iters: {args.max_iters}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")

    model.train()
    start_time = time.time()

    for iter_num in range(start_iter, args.max_iters):
        # 学习率
        lr = lr_cosine_schedule(
            iter_num, args.lr, args.min_lr, 
            args.warmup_iters, args.cosine_cycle_iters
        )
        
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 采样
        x, y = data_loader(train_data, args.batch_size, args.context_length, args.device)

        # 前向传播 + 损失计算
        logits = model(x)
        logits = rearrange(logits, '... vocab_size -> (...) vocab_size')
        loss = cross_entropy(logits, y.view(-1))

        # 反向传播 
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 梯度裁剪
        gradient_clipping(model.parameters(), args.grad_clip)

        # 更新参数
        optimizer.step()

        # log
        if iter_num % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"iter {iter_num}: loss={loss.item():.4f}, lr={lr:.2e}, elapsed={elapsed:.1f}s"
            )
            writer.writerow([iter_num, loss.item(), "", lr, elapsed])
            log_file.flush()

        # 评估
        if iter_num > 0 and iter_num % args.eval_interval == 0:
            if valid_data is not None:
                val_loss = evaluate(model, valid_data, args)
                elapsed = time.time() - start_time
                print(f"iter {iter_num}: val_loss={val_loss:.4f}")
                writer.writerow([iter_num, "", val_loss, lr, elapsed])
                log_file.flush()
            save_checkpoint(model, optimizer, iter_num, args.checkpoint_path)
            print(f"Saved checkpoint to {args.checkpoint_path}")

    # 保存 Checkpoint
    save_checkpoint(model, optimizer, args.max_iters, args.checkpoint_path)
    print(f"Final checkpoint saved to {args.checkpoint_path}")

    print("\nTraining complete!")
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.1f}s")

    log_file.close()

    # 画图
    try:
        plot_path = os.path.join(log_dir, "training_curves.png")
        plot_training_log(log_path, plot_path)
    except ImportError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()