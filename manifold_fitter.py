import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import process_landmarks
import yaml
from sklearn.model_selection import train_test_split


class ScaledBasis(nn.Module):
    def __init__(self, init_scales=(.5, 0.5, .5)):
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
        self.true_loss = True

        # learnable params
        self.basis = ScaledBasis().to(device)
        self.s0 = nn.Parameter(torch.randn(3, device=device))

        self.params = [self.s0] + list(self.basis.parameters())

    def residuals(self, u, v, l):
        phi, psi, n = self.basis()
        a = u[:, None] * phi[None, :] + v[:, None] * psi[None, :] + self.s0[None, :]
        if not self.true_loss:
            # Compute cross product
            crosses = torch.cross(a, l, dim=1)       # (N,3)
            norms = torch.norm(crosses, dim=1)  # L2 norm per row
            return (norms ** 2).mean() + 1 / (psi.norm() ** 2 + phi.norm() ** 2 + n.norm() ** 2)  # or torch.sum(norms) if you want sum
        else:
            nl = torch.einsum("ni,i->n", l, n)
            ns = torch.dot(n, self.s0)
            return (torch.norm(a - l * ns / nl[:, None]) ** 2).mean() + 1 / (psi.norm() ** 2 + phi.norm() ** 2 + n.norm() ** 2) 
    
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


    def run(self, val_split=0.25):
        # Load and convert to torch
        u, v, l = self.load_state()

        # Split into training and validation sets
        u_train, u_val, v_train, v_val, l_train, l_val = train_test_split(
            u.cpu().numpy(), v.cpu().numpy(), l.cpu().numpy(),
            test_size=val_split, random_state=42
        )
        u_train, v_train, l_train = map(lambda x: torch.tensor(x, dtype=torch.float32, device=self.device), (u_train, v_train, l_train))
        u_val, v_val, l_val = map(lambda x: torch.tensor(x, dtype=torch.float32, device=self.device), (u_val, v_val, l_val))

        best_val_loss = float('inf')
        opt = optim.Adam(self.params, lr=self.lr)

        for step in tqdm(range(self.steps)):
            # --- training step ---
            opt.zero_grad()
            train_res = self.residuals(u_train, v_train, l_train)
            train_loss = (train_res)
            train_loss.backward()
            opt.step()

            # --- validation evaluation ---
            with torch.no_grad():
                val_res = self.residuals(u_val, v_val, l_val)
                val_loss = (val_res).item()

            tqdm.write(f"Step {step}, Train Loss: {train_loss.item():.6e}, Val Loss: {val_loss:.6e}")

            # Save best parameters based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                phi, psi, n = self.basis()
                best_params = {
                    "phi": phi.detach().cpu().numpy(),
                    "psi": psi.detach().cpu().numpy(),
                    "n": n.detach().cpu().numpy(),
                    "s0": self.s0.detach().cpu().numpy(),
                }

        # Save to YAML
        print("DONE")
        print(best_params)
        with open(f"{self.calibration_dir}/calibrated_parameters.yaml", 'w') as f:
            yaml.dump(best_params, f)

        return best_params


if __name__ == "__main__":
    calib_folder = "calibration/stas/"
    fitter = ManifoldFitter(calib_folder, lr=3e-4, steps=20000)
    fitter.run()