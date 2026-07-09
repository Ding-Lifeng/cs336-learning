from jaxtyping import Float, Int, Bool
import torch
from torch import Tensor
from collections.abc import Iterable
from einops import einsum

# softmax函数
def softmax(
    in_features: Float[Tensor, " ..."], 
    dim: int
) -> Float[Tensor, " ..."]:
    # 沿 dim 维度取最大值
    max_vals = in_features.max(dim = dim, keepdim = True).values

    # 数值稳定
    shifted = in_features - max_vals

    exp_shifted = shifted.exp()

    # 归一化
    return exp_shifted / exp_shifted.sum(dim = dim, keepdim=True)

# cross_entropy_loss
def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    # 数值稳定
    max_vals = inputs.max(dim = -1, keepdim = True).values
    shifted = inputs - max_vals

    # log-sum-exp
    log_sum_up = shifted.exp().sum(dim = -1).log() + max_vals.squeeze(-1)

    # 获取预测的target logits
    predict_target_logits = inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # 使用 log-sum-exp 计算多分类cross entropy
    losses = log_sum_up - predict_target_logits
    return losses.mean()

# 梯度裁剪
def gradient_clipping(
        parameters: Iterable[torch.nn.Parameter], 
        max_l2_norm: float
) -> None:
    # 收集参数梯度
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    
    # 计算 L2 范数
    total_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), p=2)

    # 缩放
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)

# scaled_dot_product_attention
def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None 
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.size(-1)

    # 缩放系数
    scale = 1.0 / (d_k ** 0.5)

    # Q @ K^T * scale shape(..., queries keys)
    scores = einsum(Q, K, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k") * scale

    # mask
    scores = scores.masked_fill(~mask, float("-inf"))

    # softmax
    weights = softmax(scores, dim=-1)

    return einsum(weights, V, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")