import torch

def simulate_n_bit_quantization(weight, bit=4, group_size=None):
    """
    灵活支持 N-bit 的伪量化函数，用于后门防御消融实验
    """
    if bit < 2:
        raise ValueError("Bit width must be at least 2")

    # 根据 bit 动态计算表示范围
    q_min = -(2 ** (bit - 1)) * 1.0
    q_max = (2 ** (bit - 1) - 1) * 1.0

    amax = torch.amax(torch.abs(weight), dim=1, keepdim=True)

    scale = amax / q_max
    scale = torch.clamp(scale, min=1e-8)

    w_q = torch.round(weight / scale)
    w_q = torch.clamp(w_q, q_min, q_max)
    w_simulated = w_q * scale

    return w_simulated