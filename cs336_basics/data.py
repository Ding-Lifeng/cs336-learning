import torch
import numpy as np
import numpy.typing as npt

def data_loader(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # 随机采样 batch_size 个起始位置
    max_start = len(dataset) - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)

    # 读取context_length + 1个连续token，其中多出token负责next token prediction
    # shape:(batch_size, context_length + 1)
    batch = np.stack([dataset[s : s + context_length + 1] for s in starts])

    # inputs / labels
    inputs = batch[:, :context_length] # 前cotext_length个token
    labels = batch[:, 1:] # 后context_length个token

    inputs = torch.from_numpy(inputs).long().to(device)
    labels = torch.from_numpy(labels).long().to(device)

    return inputs, labels