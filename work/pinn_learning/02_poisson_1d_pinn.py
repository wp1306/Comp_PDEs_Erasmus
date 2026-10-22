"""
Step 2: PINN for a 1D Poisson (elliptic BVP)

  -u''(x) = pi^2 sin(pi x),  x in (0, 1)
  u(0) = u(1) = 0
  exact: u(x) = sin(pi x)

Loss = MSE(PDE residual) + W_BC * MSE(Dirichlet on boundaries)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pinn_utils import MLP, grad, relative_l2, set_seed

set_seed(42)
device = torch.device("cpu")

PI = np.pi
X_MIN, X_MAX = 0.0, 1.0
N_COLLOC = 256
EPOCHS = 8000
LR = 1e-3
W_BC = 200.0


def exact_u(x: np.ndarray) -> np.ndarray:
    return np.sin(PI * x)


def source(x: torch.Tensor) -> torch.Tensor:
    """f in -u'' = f, so f = pi^2 sin(pi x)."""
    return (PI**2) * torch.sin(PI * x)


def finite_difference_poisson(n: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """Simple second-order FD for -u'' = pi^2 sin(pi x), u(0)=u(1)=0."""
    x = np.linspace(X_MIN, X_MAX, n)
    h = x[1] - x[0]
    main = 2.0 / h**2
    off = -1.0 / h**2
    A = np.zeros((n - 2, n - 2))
    np.fill_diagonal(A, main)
    np.fill_diagonal(A[1:], off)
    np.fill_diagonal(A[:, 1:], off)
    f = (PI**2) * np.sin(PI * x[1:-1])
    u_inner = np.linalg.solve(A, f)
    u = np.zeros(n)
    u[1:-1] = u_inner
    return x, u


def train_pinn() -> tuple[MLP, list[float]]:
    model = MLP(d_in=1, d_out=1, width=64, depth=5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    x_int = torch.linspace(X_MIN + 1e-3, X_MAX - 1e-3, N_COLLOC, device=device).view(
        -1, 1
    )
    x_int.requires_grad_(True)

    x_bc = torch.tensor([[X_MIN], [X_MAX]], device=device, dtype=torch.float32)
    u_bc_target = torch.zeros(2, 1, device=device)

    history: list[float] = []
    for epoch in range(EPOCHS):
        opt.zero_grad()

        u_int = model(x_int)
        du_dx = grad(u_int, x_int)
        d2u_dx2 = grad(du_dx, x_int)
        res = -d2u_dx2 - source(x_int)  # -u'' - f = 0  <=>  -u'' = f
        loss_pde = torch.mean(res**2)

        u_bc = model(x_bc)
        loss_bc = torch.mean((u_bc - u_bc_target) ** 2)

        loss = loss_pde + W_BC * loss_bc
        loss.backward()
        opt.step()
        history.append(loss.item())

        if (epoch + 1) % 2000 == 0:
            print(
                f"epoch {epoch+1:5d}  loss={loss.item():.3e}  "
                f"pde={loss_pde.item():.3e}  bc={loss_bc.item():.3e}"
            )

    return model, history


def main() -> None:
    model, history = train_pinn()

    x_test = np.linspace(X_MIN, X_MAX, 400)
    x_t = torch.tensor(x_test, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        u_pinn = model(x_t).cpu().numpy().ravel()
    u_exact = exact_u(x_test)
    err_pinn = relative_l2(u_pinn, u_exact)

    x_fd, u_fd = finite_difference_poisson()
    u_fd_interp = np.interp(x_test, x_fd, u_fd)
    err_fd = relative_l2(u_fd_interp, u_exact)

    print(f"relative L2 error  PINN: {err_pinn:.4e}")
    print(f"relative L2 error  FD:   {err_fd:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(history)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("training loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_test, u_exact, "k-", lw=2, label="exact sin(pi x)")
    axes[1].plot(x_test, u_pinn, "r--", lw=1.5, label=f"PINN ({err_pinn:.1e})")
    axes[1].plot(x_fd, u_fd, "b:", lw=2, label=f"FD ({err_fd:.1e})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u")
    axes[1].set_title("1D Poisson BVP")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = __file__.replace(".py", "_result.png")
    fig.savefig(out, dpi=120)
    print(f"saved figure: {out}")
    plt.close()


if __name__ == "__main__":
    main()
