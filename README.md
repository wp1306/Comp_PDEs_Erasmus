# Computational PDEs: Classical Approaches and Physics-Informed Learning

Course materials for an **Erasmus+ ICM Teaching Mobility** at  
**Eötvös Loránd University (ELTE), Faculty of Informatics**.

Neptun title: **PDEs and Physics-Informed Neural Networks**

Repository: [github.com/wp1306/Comp_PDEs_Erasmus](https://github.com/wp1306/Comp_PDEs_Erasmus)

---

## Instructor

| | |
|---|---|
| **Name** | Hiroyuki Miyoshi（三好 裕之） |
| **Affiliation** | The University of Tokyo |
| **GitHub** | [wp1306](https://github.com/wp1306) |
| **Academic coordinator (ELTE)** | Prof. Viktória Zsók (`zsv@inf.elte.hu`) |

---

## Practical information

| | |
|---|---|
| **Host** | ELTE Faculty of Informatics, South Building (*Déli Tömb*), Pázmány Péter sétány 1/C |
| **Dates** | 9–12 November 2026 (Mon–Thu) |
| **Time** | 8:15–10:00 each morning (105 minutes, no break; room booked 8:00–10:00) |
| **Language** | English |
| **Audience** | Informatics MSc (two tracks); PhD students may also attend |

### Rooms

| Day | Date | Room |
|-----|------|------|
| Monday | 9 Nov 2026 | **0-412** Rényi Alfréd terem (interactive board, 21 seats) |
| Tuesday | 10 Nov 2026 | **0-412** |
| Wednesday | 11 Nov 2026 | **0-823** Kitaibel Pál terem (projector only, 200 seats) |
| Thursday | 12 Nov 2026 | **0-412** |

Wednesday has a projector and no board: figures are prepared in advance; long live coding is not planned that morning.

---

## Course overview

Differential equations change with the **operator**, the **domain**, and the **boundary constraints**.  
This short course uses elliptic problems (Laplace / Poisson as the main examples) and compares two programmes for the *same* equation, domain, and constraints:

1. **Classical** — exact up to a truncation you choose (Fourier series, method of images, infinite products on circular domains).
2. **Physics-informed learning (PINNs)** — a small network trained so that residuals of the equation (and of functional identities) are small.

A small training loss is **not** automatically a correct solution; answers can depend on initialisation. That contrast becomes sharper when the domain has **holes**: each new boundary circle adds a constraint for both the product and the network.

**Complex analysis** enters as the classical toolkit for circular boundaries (Cauchy–Riemann, images, conformal maps, the Schottky–Klein prime function). A point vortex is only a **test case** that makes the boundary condition visible—not a course on fluid dynamics.

The pedagogical pipeline each morning is:

**problem → algorithm → code → check**

The course sits next to ELTE offerings such as *Topics in Applied Mathematics*, *Numerical Methods for Optimization*, and *Methods and Tools for AI Applications*. Prior complex analysis or fluid dynamics is **not** assumed.

A short syllabus PDF is in [`contents/syllabus20261016.pdf`](contents/syllabus20261016.pdf).

---

## Four mornings

| Day | Theme | Network? |
|-----|--------|----------|
| **Mon 9** | ODE / Poisson: Fourier (and FD ideas) vs PINN; relative $L^2$; warm start | Yes (stored runs) |
| **Tue 10** | Harmonic functions, Cauchy–Riemann, images on half-plane / disk; conformal maps (simply connected) | Demo only if time (e.g. disk → ellipse) |
| **Wed 11** | Multiply connected domains: annulus, product $P(\zeta,\rho)$, identities, point vortex | No |
| **Thu 12** | Same identities as a **loss**; recover $P$ with a network; compare to the truncated product; vortex from $P$ and from $P_\theta$ | Yes (stored runs) |

Training in class uses **precomputed** notebook runs (Monday and Thursday). Live long training is not required.

---

## Prerequisites

**Assumed:** Python (NumPy, Matplotlib); linear systems / least squares; the words *derivative*, *boundary condition*, *residual*, *loss*.

**Not assumed:** prior PyTorch; complex analysis; fluid dynamics.

---

## Notebooks and setup

- **Environment guide:** [`work/pinn_learning/ENV_SETUP.md`](work/pinn_learning/ENV_SETUP.md)  
  (VS Code + conda/venv; environment name `comp_pde`)
- **Dependencies:** [`work/pinn_learning/requirements-comp_pde.txt`](work/pinn_learning/requirements-comp_pde.txt)
- **Start here:** [`work/pinn_learning/01_ode_pinn.ipynb`](work/pinn_learning/01_ode_pinn.ipynb)

Suggested order:

| Notebook | Role |
|----------|------|
| `01_ode_pinn.ipynb` (+ variants) | ODE residual, autograd, PINN |
| `02_poisson_1d_pinn.ipynb` | 1D Poisson: FD + PINN |
| `03_poisson_2d_pinn.ipynb` | 2D Poisson PINN |
| `04_poisson_2d_pinn_warmstart.ipynb` | Warm start (optional) |
| `03_schottky_klein_prime_nn.ipynb` | Prime-function network (Day 4) |

Shared helpers: [`work/pinn_learning/pinn_utils.py`](work/pinn_learning/pinn_utils.py).

---

## Repository layout

| Path | Description |
|------|-------------|
| `work/pinn_learning/` | Jupyter notebooks, `pinn_utils.py`, environment files |
| `contents/` | Syllabus / handouts (PDF) |

---

## Selected references

- M. Raissi, P. Perdikaris, G. E. Karniadakis (2019), *J. Comput. Phys.* — PINNs  
- D. Crowdy (2020), *Solving Problems in Multiply Connected Domains*, SIAM — prime functions  
- S. Cuomo et al. (2022), *J. Sci. Comput.* — survey on physics-informed learning  

See the syllabus PDF for the full short bibliography.
