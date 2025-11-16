import torch


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (SAM) optimizer wrapper.

    This implementation wraps a base optimizer (e.g., SGD) and performs
    two-step updates: ascent to the neighborhood boundary, then descent
    using the gradient at the perturbed weights.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False):
        if rho <= 0:
            raise ValueError("Invalid rho, should be > 0")

        defaults = dict(rho=rho, adaptive=adaptive)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **{})
        self.rho = rho
        self.adaptive = adaptive

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = self.state[p].pop("e_w", None)
                if e_w is not None:
                    p.sub_(e_w)

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        raise RuntimeError("SAM step requires explicit first_step and second_step calls")

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

