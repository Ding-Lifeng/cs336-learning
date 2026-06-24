import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from einops import einsum, rearrange

# 线性变换 - 矩阵乘法
class Linear(nn.Module):
    def __init__(self, in_features_dim, out_features_dim, device=None, dtype=None):
        super().__init__()
        
        # 构造权重矩阵
        self.weight = nn.Parameter(
            torch.empty(out_features_dim, in_features_dim, device=device, dtype=dtype)
        )

        # 初始化权重矩阵 Truncated Normal
        std = (2.0 / (in_features_dim + out_features_dim)) ** 0.5 # 初始化标准差
        nn.init.trunc_normal_(self.weight, std=std, a=-2*std, b= -2*std)

    def forward(
        self, 
        in_features: Float[Tensor, " ... d_model"]
        ) -> Float[Tensor, " ... d_out"]:
        # in: (..., in_features)
        # output: (..., out_features)
        return einsum(in_features, self.weight, "... d_in, d_out d_in -> ... d_out")

# def linear(
#     d_in: int,
#     d_out: int,
#     weights: Float[Tensor, " d_out d_in"],
#     in_features: Float[Tensor, " ... d_in"],   
# ) -> Float[Tensor, " ... d_out"]:
#     return einsum(in_features, weights, "... d_in, d_out d_in -> ... d_out")

# 词嵌入 - 查表
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        nn.init.trunc_normal_(self.weight, std=1.0, a=-2, b=2)

    def forward(
        self, 
        token_ids: Int[Tensor, " ..."],
        ) -> Float[Tensor, "... d_model"]:
        return self.weight[token_ids]

# def embedding(
#     vocab_size: int,
#     d_model: int,
#     weights: Float[Tensor, " vocab_size d_model"],
#     token_ids: Int[Tensor, " ..."],
# ) -> Float[Tensor, "... d_model"]:
#     # one-hot 编码
#     one_hot = torch.nn.functional.one_hot(token_ids, vocab_size).float()

#     # 矩阵乘法
#     return einsum(one_hot, weights, " ... vocab_size, vocab_size d_model -> ... d_model")

# silu函数
def silu(
    in_features: Float[Tensor, " ..."]
) -> Float[Tensor, " ..."]:
    return in_features * torch.sigmoid(in_features) 

# 均方根层归一化
class Rmsnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(d_model, device = device, dtype = dtype))

        self.eps = eps

    def forward(
        self, 
        in_features: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_model"]:
        original_dtype = in_features.dtype

        # upcast float32
        in_features_fp32 = in_features.to(torch.float32)

        # 计算 root mean square
        mean_square = (in_features_fp32 ** 2).mean(dim = -1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + self.eps)

        # 缩放
        normalized = in_features_fp32 * inv_rms
        result = einsum(normalized, self.weight, "... d_model, d_model -> ... d_model")
        
        # downcast
        return result.to(original_dtype)

# def rmsnorm(
#     d_model: int,
#     eps: float,
#     weights: Float[Tensor, " d_model"],
#     in_features: Float[Tensor, " ... d_model"],
# ) -> Float[Tensor, " ... d_model"]:
#     # 计算 root mean square
#     mean_square = (in_features ** 2).mean(dim=-1, keepdim=True)
#     inv_rms = torch.rsqrt(mean_square + eps)

#     # 缩放
#     normalized = in_features * inv_rms 
#     return einsum(normalized, weights, "... d_model, d_model -> ... d_model")

# SwiGLU Swish-Gated Linear Unit (前馈网络FFN优化方案)
class SwiGLU(nn.Module):
    def __init__(self, d_model_dim, d_ff_dim, device=None, dtype=None):
        super().__init__()
        # silu激活 
        # x @ w1.T -> (..., d_ff)
        self.w1 = Linear(d_model_dim, d_ff_dim, device=device, dtype=dtype)
        
        # 原始信息
        # x @ w3.T -> (..., d_ff)
        self.w3 = Linear(d_model_dim, d_ff_dim, device=device, dtype=dtype)
        
        # 输出投影
        self.w2 = Linear(d_ff_dim, d_model_dim, device=device, dtype=dtype)

    def forward(self, in_features):
        # (..., d_model)
        gated = silu(self.w1(in_features)) * self.w3(in_features)
        return self.w2(gated)

# def swiglu(
#     d_model: int,
#     d_ff: int,
#     w1_weight: Float[Tensor, " d_ff d_model"],
#     w2_weight: Float[Tensor, " d_model d_ff"],
#     w3_weight: Float[Tensor, " d_ff d_model"],
#     in_features: Float[Tensor, " ... d_model"],
# ) -> Float[Tensor, "... d_model"]:
#     # silu激活 
#     # x @ w1.T -> (..., d_ff)
#     # a = einsum(in_features, w1_weight, "... d_model, d_ff d_model -> ..., d_ff")
#     a = linear(d_model, d_ff, w1_weight, in_features)

#     # 原始信息
#     # x @ w3.T -> (..., d_ff)
#     # b = einsum(in_features, w3_weight, "... d_model, d_ff d_model -> ..., d_ff")
#     b = linear(d_model, d_ff, w3_weight, in_features)

#     # 逐元素相乘
#     gated = silu(a) * b

#     # 输出投影
#     # return einsum(gated, w2_weight, "... d_ff, d_model d_ff -> ... d_model")
#     return linear(d_ff, d_model, w2_weight, gated)

# RoPE Rotary Positional Embedding 旋转位置嵌入
class RoPE(nn.Module):
    def __init__(
        self, 
        d_k: int,
        theta: float,
        max_seq_len: int,
        device = None
    ):
        super().__init__()
    
        # k = 1, 2, ..., d_k/2
        k = torch.arange(1, d_k // 2 + 1, device=device, dtype=torch.float32)
        # 计算theta_k 
        # shape:(d_k / 2, )
        theta_k = theta ** (-(2 * k -2) / d_k)

        # 位置索引
        # shape:(max_seq_len, )
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32) 
        # theta_i,k = i / theta^((2k-2)/d_k)      
        # # shape:(max_seq_len, d_k / 2)  
        m_theta = positions.unsqueeze(1) * theta_k.unsqueeze(0)

        # 计算 cos 和 sin
        self.register_buffer("cos", torch.cos(m_theta))
        self.register_buffer("sin", torch.sin(m_theta))

    def forward(
        self,
        input: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # even dim
        input_even = input[..., 0::2]

        # odd dim
        input_odd = input[..., 1::2]

        # 旋转
        # even: x_2k = x_2k * cos - x_2k+1 * sin
        # odd: x_2k+1 = x_2k * sin + x_2k+1 * cos
        input_rot_even = input_even * cos - input_odd * sin
        input_rot_odd = input_even * sin + input_odd * cos

        # 交错合并
        output = torch.stack([input_rot_even, input_rot_odd], dim=-1)
        return output.flatten(-2)
    
from cs336_basics.nn_utils import scaled_dot_product_attention

# 多头自注意力机制
class MultiheadSelfAttention(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        rope = None,
        device = None,
        dtype = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # d_model必须是num_heads的整数倍
        assert d_model % num_heads == 0

        self.d_head = d_model // num_heads

        # Linear 投影
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # RoPE 模块
        self.rope = rope

    def forward(
        self, 
        in_features: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None
        ) -> Float[Tensor, " ... sequence_length d_model"]:
            # 计算Q/K/V
            Q = self.q_proj(in_features)
            K = self.k_proj(in_features)
            V = self.v_proj(in_features)

            # 分割多头(..., seq_len, d_model) -> (..., num_heads, seq_len, d_model) - 分割特征维度
            Q = rearrange(Q, "... seq (head d_head) -> ... head seq d_head", head=self.num_heads)
            K = rearrange(K, "... seq (head d_head) -> ... head seq d_head", head=self.num_heads)
            V = rearrange(V, "... seq (head d_head) -> ... head seq d_head", head=self.num_heads)
            
            # RoPE 可选
            if self.rope is not None and token_positions is not None:
                Q = self.rope(Q, token_positions)
                K = self.rope(K, token_positions)

            # causal mask 训练阶段防止模型使用未来token作弊
            seq_len = in_features.size(-2)
            mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device)
            )

            # attention
            attn = scaled_dot_product_attention(Q, K, V, mask)

            # 合并
            attn = rearrange(attn, "... head seq d_head -> ... seq (head d_head)")

            # 输出投影
            return self.o_proj(attn)
    
# TransformerBlock
# in_features -> RMSNorm -> MultiHeadSelfAttention(RoPE) -> residual -> RMSNorm -> SwiGLU -> residual -> output
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None
    ):
        super().__init__()

        # 第一层 RMSNorm
        self.ln1 = Rmsnorm(d_model=d_model, device=device, dtype=dtype)
        
        # RoPE
        rope = RoPE(theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len, device=device)

        # MHA
        self.attn = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, rope=rope, device=device, dtype=dtype)

        # 第二层 RMSNorm
        self.ln2 = Rmsnorm(d_model=d_model, device=device, dtype=dtype)

        # SwiGLU FFN
        self.ffn = SwiGLU(d_model_dim=d_model, d_ff_dim=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        in_features: Float[Tensor, " batch sequence_length d_model"],
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        # 构造 token_positions
        seq_len = in_features.size(-2)
        token_positions = torch.arange(seq_len, device=in_features.device)

        # Pre-Norm + MHA + residual
        attn_out = self.attn(self.ln1(in_features), token_positions)
        re1 = in_features + attn_out

        # Pre-Norm + FFN + residual
        ffn_out = self.ffn(self.ln2(re1))
        return re1 + ffn_out    
    
# TransformerLM
# Token embedding -> Transformer Block * n -> Norm -> Linear -> softmax
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device = None,
        dtype = None
    ):
        super().__init__()

        # Token embedding
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)

        # Transformer Block * n
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype)
            for _ in range(num_layers)
        ])

        # Norm
        self.ln_final = Rmsnorm(d_model=d_model, device=device, dtype=dtype)

        # Linear
        self.lm_head = Linear(d_model, vocab_size, device, dtype)
    
    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"],
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        # Token embedding
        output = self.token_embeddings(in_indices)

        # Transformer Block * n
        for layer in self.layers:
            output = layer(output)

        # Norm
        output = self.ln_final(output)

        # Linear
        return self.lm_head(output)