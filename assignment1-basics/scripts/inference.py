import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import softmax

DEFAULT_VOCAB_PATH = "data/tokenizers/tinystories_vocab_10000.json"
DEFAULT_MERGES_PATH = "data/tokenizers/tinystories_merges_10000.json"
DEFAULT_CHECKPOINT = "checkpoints/transformer_lm_lr1e-3_bs32_10000.pt"

DEFAULT_VOCAB_SIZE = 10000
DEFAULT_D_MODEL = 512
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_HEADS = 16
DEFAULT_D_FF = 1344
DEFAULT_CONTEXT_LENGTH = 256
DEFAULT_ROPE_THETA = 10000.0

DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from TransformerLM")
    parser.add_argument("--prompt", type=str, default="Once upon a time.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.8, help="1.0 不使用nucleus threshold")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)

    return parser.parse_args()

def load_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}...")
    model = TransformerLM(
        vocab_size=DEFAULT_VOCAB_SIZE,
        context_length=DEFAULT_CONTEXT_LENGTH,
        d_model=DEFAULT_D_MODEL,
        num_layers=DEFAULT_NUM_LAYERS,
        num_heads=DEFAULT_NUM_HEADS,
        d_ff=DEFAULT_D_FF,
        rope_theta=DEFAULT_ROPE_THETA,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    
    model.eval()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def load_tokenizer() -> Tokenizer:
    return Tokenizer.from_files(
        vocab_filepath=DEFAULT_VOCAB_PATH,
        merges_filepath=DEFAULT_MERGES_PATH,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
    )

def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    sorted_probs = softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)

    mask = cumprobs - sorted_probs > top_p
    sorted_logits[mask] = float("-inf")

    filtered = torch.full_like(logits, float("-inf"))
    filtered[sorted_idx] = sorted_logits
    return filtered

@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_id: int | None,
    device: str,
) -> list[int]:
    # 自回归生成 sample one token at a time
    generated = list(prompt_ids)

    for step in range(max_new_tokens):
        # 截取最近 DEFAULT_CONTEXT_LENGTH Token
        context = generated[-DEFAULT_CONTEXT_LENGTH:]
        x = torch.tensor([context], dtype=torch.long, device=device)

        # 前向传播 计算logits
        logits = model(x)[0, -1, :] # 取最后一个位置

        # Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Top-p sampling
        logits = top_p_filter(logits, top_p)

        # Sample
        probs = softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_id)

        if eos_id is not None and next_id == eos_id:
            print(f"[EOS at step {step}]")
            break

    return generated

def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model = load_model(args.checkpoint, args.device)
    tokenizer = load_tokenizer()

    # EOS token id
    eos_bytes = DEFAULT_SPECIAL_TOKENS[0].encode("utf-8")
    eos_id = tokenizer.bytes_to_id.get(eos_bytes)

    # Encode prompt
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"\nPrompt: {args.prompt!r}")
    print(f"Prompt tokens ({len(prompt_ids)}): {prompt_ids[:20]}...")
    print(f"Generating up to {args.max_new_tokens} tokens")

    # Generate
    output_ids = generate(
        model, prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_id=eos_id,
        device=args.device,
    )

    # Decode
    output_text = tokenizer.decode(output_ids)

    print("\n" + "=" * 60)
    print("Generated text:")
    print("=" * 60)
    print(output_text)
    print("=" * 60)

if __name__ == "__main__":
    main()                                  