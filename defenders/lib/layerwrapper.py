import torch
import torch.nn as nn

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples



class VarWrappedGPT:
    """
    This class wraps a GPT layer for specific operations,
    added variance calculation for backdoor extraction.
    """
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev) # E[X^2]
        self.mean_row = torch.zeros((self.columns), device=self.dev)   # E[X]
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t() # shape: (hidden_dim, batch*seqlen)

        # 更新 E[X^2] 的逻辑保留原来 Wanda 的方式
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.mean_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        # scaler_row 记录 L2_norm^2 的平均即 E[X^2]
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
        # mean_row 记录绝对值之和的平均即 E[|X|]
        self.mean_row += torch.sum(torch.abs(inp), dim=1) / self.nsamples

    def get_metric(self):
        """返回 均方根(激活大小) 和 方差"""
        # RMS
        rms = torch.sqrt(self.scaler_row)
        # Var(X) = E[X^2] - (E[|X|])^2
        variance = self.scaler_row - (self.mean_row ** 2)
        variance = torch.clamp(variance, min=1e-8)
        return rms, variance