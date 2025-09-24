import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import process_landmarks
import yaml

class ScaledBasis(nn.Module):
    def __init__(self, init_scales=(.05, 0.05, .1)):
        super().__init__()
        self.q = nn.Parameter(torch.tensor((0., 0., 0., 1.)))  # quaternion for rotation
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
    def __init__(self, calibration_dir: str, lr=1e-2, steps=2000, device="cpu"):
        self.device = device
        self.lr = lr
        self.steps = steps
        self.calibration_dir = calibration_dir

        # learnable params
        self.basis = ScaledBasis().to(device)
        self.s0 = nn.Parameter(torch.randn(3, device=device))

        self.params = [self.s0] + list(self.basis.parameters())

    def residuals(self, u, v, l):
        phi, psi, n = self.basis()
        a = u[:, None] * phi[None, :] + v[:, None] * psi[None, :] + self.s0[None, :]
        # Compute cross product
        crosses = torch.cross(a, l, dim=1)       # (N,3)
        norms = torch.norm(crosses, dim=1)  # L2 norm per row
        return norms  # or torch.sum(norms) if you want sum
    
    def get_ray_directions(self, landmarks):
        return torch.Tensor([process_landmarks(landmark.numpy())[1] for landmark in landmarks], device=self.device)

    def load_state(self):
        u, v, landmarks = [np.load(f"{self.calibration_dir}/{name}.npy") for name in ('u', 'v', 'landmarks')]

        u = torch.tensor(u, dtype=torch.float32, device=self.device)
        v = torch.tensor(v, dtype=torch.float32, device=self.device)
        landmarks = torch.tensor(landmarks, dtype=torch.float32, device=self.device)
        
        l = self.get_ray_directions(landmarks)
        #print(l)

        return u, v, l


    def run(self):
        # convert to torch
        u, v, l = self.load_state()

        opt = optim.Adam(self.params, lr=self.lr)
        for step in tqdm(range(self.steps)):
            opt.zero_grad()
            res = self.residuals(u, v, l)
            loss = (res**2).mean()
            tqdm.write(f"{loss.item()}")
            loss.backward()
            opt.step()

        # extract learned constants
        phi, psi, n = self.basis()
        res = {
            "phi": phi.detach().cpu().numpy(),
            "psi": psi.detach().cpu().numpy(),
            "n": n.detach().cpu().numpy(),
            "s0": self.s0.detach().cpu().numpy(),
        }
        print("DONE")
        print(res)
        with open(f"{self.calibration_dir}/calibrated_parameters.yaml", 'w') as f:
            yaml.dump(res, f)
        return res


if __name__ == "__main__":
    calib_folder = "calibration/daniil/"
    fitter = ManifoldFitter(calib_folder, 3e-4, 20000)
    fitter.run()

