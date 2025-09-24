import torch
import torch.nn as nn
import torch.optim as optim

class ScaledBasis(nn.Module):
    def __init__(self, init_scales=(1.0, 1.0, 1.0)):
        super().__init__()
        self.q = nn.Parameter(torch.randn(4))  # quaternion for rotation
        self.scales = nn.Parameter(torch.tensor(init_scales, dtype=torch.float))

    def forward(self):
        # normalize quaternion
        q = self.q / self.q.norm()
        w, x, y, z = q
        R = torch.stack([
            torch.stack([1-2*(y**2+z**2), 2*(x*y - z*w),   2*(x*z+y*w)]),
            torch.stack([2*(x*y+z*w),     1-2*(x**2+z**2), 2*(y*z-x*w)]),
            torch.stack([2*(x*z-y*w),     2*(y*z+x*w),     1-2*(x**2+y**2)])
        ])
        phi = self.scales[0] * R[:, 0]
        psi = self.scales[1] * R[:, 1]
        n   = self.scales[2] * R[:, 2]
        return phi, psi, n

class ManifoldFitter:
    def __init__(self, lr=1e-2, steps=2000, device="cpu"):
        self.device = device
        self.lr = lr
        self.steps = steps

        # learnable params
        self.basis = ScaledBasis().to(device)
        self.s0 = nn.Parameter(torch.randn(3, device=device))

        self.params = [self.s0, self.c] + list(self.basis.parameters())

    def residuals(self, u, v, l):
        phi, psi, n = self.basis()
        a = u[:, None] * phi[None, :] + v[:, None] * psi[None, :] + self.s0[None, :]
        # Compute cross product
        crosses = torch.cross(a, l)       # (N,3)
        norms = torch.norm(crosses, dim=1)  # L2 norm per row
        return norms.mean()  # or torch.sum(norms) if you want sum


    def run(self, u, v, l):
        # convert to torch
        u = torch.tensor(u, dtype=torch.float32, device=self.device)
        v = torch.tensor(v, dtype=torch.float32, device=self.device)
        l = torch.tensor(l, dtype=torch.float32, device=self.device)

        opt = optim.Adam(self.params, lr=self.lr)
        for step in range(self.steps):
            opt.zero_grad()
            res = self.residuals(u, v, l)
            loss = (res**2).mean()
            loss.backward()
            opt.step()

        # extract learned constants
        phi, psi, n = self.basis()
        return {
            "phi": phi.detach().cpu().numpy(),
            "psi": psi.detach().cpu().numpy(),
            "n": n.detach().cpu().numpy(),
            "s0": self.s0.detach().cpu().numpy(),
            "c": self.c.detach().cpu().numpy(),
        }


def main():
    file_path = "calib.yaml"
    calib_folder = ""
    fitter = ManifoldFitter(3e-4, 2000)
    u, v, l = ...
    fitter.run()