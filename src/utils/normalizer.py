"""Running mean/std normalizer using Welford's online algorithm."""
import os
import torch


class RunningNormalizer:
    def __init__(self, dim: int, eps: float = 1e-8, device: str = "cpu"):
        self.dim = dim
        self.eps = eps
        self.device = device
        self.mean = torch.zeros(dim, dtype=torch.float32)
        self.var = torch.ones(dim, dtype=torch.float32)
        self.count = 0

    def update(self, x: torch.Tensor) -> None:
        x = x.float().cpu()
        B = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
            self.count = B
        else:
            new_count = self.count + B
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * B / new_count
            m_a = self.var * self.count
            m_b = batch_var * B
            new_var = (m_a + m_b + delta ** 2 * self.count * B / new_count) / new_count
            self.mean = new_mean
            self.var = new_var
            self.count = new_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device)
        std = (self.var + self.eps).sqrt().to(x.device)
        return (x - mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device)
        std = (self.var + self.eps).sqrt().to(x.device)
        return x * std + mean

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"mean": self.mean, "var": self.var, "count": self.count,
                    "dim": self.dim, "eps": self.eps}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "RunningNormalizer":
        data = torch.load(path, map_location="cpu")
        obj = cls(dim=data["dim"], eps=data["eps"], device=device)
        obj.mean = data["mean"]
        obj.var = data["var"]
        obj.count = data["count"]
        return obj
