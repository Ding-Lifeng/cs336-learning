import torch
import math

# AdamW 优化器
class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-3,
        betas = (0.9, 0.95),
        eps = 1e-8,
        weight_decay = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # 初始状态
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    state['t'] = 0

                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                lr = group['lr']
                eps = group['eps']
                wd = group['weight_decay']

                state['t'] += 1
                t = state['t']

                # adjusted 𝛼
                alpha_t = lr * ((1 -beta2 ** t) ** 0.5) / (1 - beta1 ** t)

                # 权重衰减
                p.mul_(1 - lr * wd)

                # 一阶矩 first moment estimate
                m.mul_(beta1).add_(grad, alpha = 1 - beta1)

                # 二阶矩 second moment estimate
                v.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)

                # 更新
                p.addcdiv_(m, v.sqrt() + eps, value=-alpha_t)

def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    # warm-up
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    
    # cosine annealing
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
    
    # post annealing
    return min_learning_rate