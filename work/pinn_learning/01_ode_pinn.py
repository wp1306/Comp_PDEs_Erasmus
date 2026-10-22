"""
Step 1: PINN for a simple ODE (warm-up)

  u'(x) + u(x) = 0,  x in (0, 1),  u(0) = 1
  exact: u(x) = exp(-x)

Loss = MSE(PDE residual on collocation points) + weight * MSE(BC at x=0)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pinn_utils import MLP, grad, relative_l2, set_seed

set_seed(42)
device = torch.device("cpu")

# --- problem ---
X_MIN, X_MAX = 0.0, 1.0
N_COLLOC = 200
N_BOUNDARY = 1
EPOCHS = 5000
LR = 1e-3
W_BC = 100.0  # boundary weight (often needs tuning)


def exact_u(x: np.ndarray) -> np.ndarray:
    return np.exp(-x)


def train_pinn() -> tuple[MLP, list[float]]:
    model = MLP(d_in=1, d_out=1, width=32, depth=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    x_int = torch.linspace(X_MIN + 1e-3, X_MAX, N_COLLOC, device=device).view(-1, 1)
    x_int.requires_grad_(True)
    x0 = torch.zeros(N_BOUNDARY, 1, device=device, requires_grad=True)
    u0_target = torch.ones(N_BOUNDARY, 1, device=device)

    history: list[float] = []
    for epoch in range(EPOCHS):
        opt.zero_grad()

        u_int = model(x_int)
        du_dx = grad(u_int, x_int)
        res = du_dx + u_int  # u' + u = 0
        loss_pde = torch.mean(res**2)

        u0 = model(x0)
        loss_bc = torch.mean((u0 - u0_target) ** 2)

        loss = loss_pde + W_BC * loss_bc
        loss.backward()
        opt.step()
        history.append(loss.item())

        if (epoch + 1) % 1000 == 0:
            print(
                f"epoch {epoch+1:5d}  loss={loss.item():.3e}  "
                f"pde={loss_pde.item():.3e}  bc={loss_bc.item():.3e}"
            )

    return model, history


def main() -> None:
    model, history = train_pinn()

    x_test = np.linspace(X_MIN, X_MAX, 300)
    x_t = torch.tensor(x_test, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        u_pred = model(x_t).cpu().numpy().ravel()
    u_ref = exact_u(x_test)
    err = relative_l2(u_pred, u_ref)
    print(f"relative L2 error vs exp(-x): {err:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(history)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("training loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_test, u_ref, "k-", lw=2, label="exact exp(-x)")
    axes[1].plot(x_test, u_pred, "r--", lw=1.5, label="PINN")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u")
    axes[1].set_title(f"ODE PINN (rel. L2 = {err:.2e})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = __file__.replace(".py", "_result.png")
    fig.savefig(out, dpi=120)
    print(f"saved figure: {out}")
    plt.close()


if __name__ == "__main__":
    main()
