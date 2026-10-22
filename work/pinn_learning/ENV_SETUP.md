# Python environment setup (PINN notebooks, e.g. `01_ode_pinn.ipynb`)

Use a dedicated environment named `comp_pde` (computational PDE / PINN notebooks in this folder).  
Notebooks are saved with kernel display name `comp_pde`.

Install **SciPy and SymPy** together with NumPy and PyTorch—even if you start with `01_ode_pinn.ipynb`—so `02_*`, `03_*`, … work without extra steps. See `requirements-comp_pde.txt`.

**Recommended editor (students):** [Visual Studio Code](https://code.visualstudio.com/) with the **Jupyter** and **Python** extensions (Microsoft). Course materials: [GitHub repo](https://github.com/miyoshi-hiroyuki/2026Erasmus_Hungary).

---

## Quick start (students — VS Code)

Do this once before Day 1 live coding.

| Step | Action |
| ---- | ------ |
| 1 | Install [VS Code](https://code.visualstudio.com/) |
| 2 | In VS Code, install extensions **Python** and **Jupyter** (Microsoft) |
| 3 | Install [Miniforge](https://github.com/conda-forge/miniforge) or Miniconda (optional but recommended), **or** Python 3.11 from [python.org](https://www.python.org/downloads/) for venv |
| 4 | Clone the repo: `git clone https://github.com/miyoshi-hiroyuki/2026Erasmus_Hungary.git` → open the folder in VS Code |
| 5 | Create env `comp_pde` — **§5 Option B** (conda) or **§6 Option C** (venv) |
| 6 | Open `work/pinn_learning/01_ode_pinn.ipynb` |
| 7 | **Select Kernel** → **Python Environments…** → **`comp_pde`** |
| 8 | Run all cells from the top; check `device: cpu` and the final plot |

You do **not** need a separate JupyterLab install; VS Code runs notebooks with the selected Python environment.

---

## 1. Visual Studio Code (recommended for students)

1. Install **VS Code**: https://code.visualstudio.com/ (Windows, macOS, Linux).
2. Install extensions (Extensions view, or marketplace):
   - **Python** (Microsoft)
   - **Jupyter** (Microsoft)
3. Clone or download the course repo and open the folder in VS Code (`File → Open Folder…`).
4. Open e.g. `work/pinn_learning/01_ode_pinn.ipynb`.
5. Click **Select Kernel** (top right) → **Python Environments…** → choose **`comp_pde`** (conda) or the interpreter path ending in `envs/comp_pde/bin/python`.
6. Run cells with the play button or **Shift+Enter**.

If `comp_pde` is missing from the list, finish **§5 Option B** (conda) or **§6 Option C** (venv) first, then **reload the window** (`Developer: Reload Window` from the Command Palette).

**Windows:** After `conda activate comp_pde`, pick the interpreter under `...\envs\comp_pde\python.exe`. Enable “Add Python to PATH” if you installed Python from python.org for venv.

**macOS / Linux:** Interpreter path often ends with `envs/comp_pde/bin/python`.

---

## 2. Target stack (pinned in `requirements-comp_pde.txt`)


| Component  | Version                         |
| ---------- | ------------------------------- |
| Python     | **3.11.x** (reference: 3.11.14) |
| NumPy      | **1.26.4**                      |
| Matplotlib | **3.10.9**                      |
| PyTorch    | **2.2.2** (pip)                 |
| SciPy      | **1.17.1**                      |
| SymPy      | **1.14.0**                      |
| ipykernel  | **≥ 6.29**                      |


These pins were chosen to match a stable instructor setup (legacy env `prime_mat`) so results stay comparable with `teaching/programs/recover*.ipynb` on the same machine. 

**Note:** The local `.venv` in this folder may be **Python 3.13** and incomplete. Prefer `comp_pde` with Python **3.11**.

---



## 3. Which setup to use


| Option                             | Best for            | Summary                                                              |
| ---------------------------------- | ------------------- | -------------------------------------------------------------------- |
| **A. Use existing** `comp_pde`     | Env already created | `conda activate comp_pde`, kernel `comp_pde`                         |
| **B. New conda env (recommended)** | First-time setup    | `conda create -n comp_pde python=3.11` + `requirements-comp_pde.txt` |
| **C. venv**                        | No conda            | `python3.11 -m venv` + same requirements file                        |


---



## 4. Option A — Use existing `comp_pde`

```bash
conda activate comp_pde
```

Open the notebook in **VS Code** and set **Kernel** to `comp_pde`.  
Run from the top; the first code cell should print e.g. `device: cpu` (or MPS/CUDA).

```bash
conda activate comp_pde
python -c "import numpy, scipy, sympy, torch, matplotlib; print(numpy.__version__, scipy.__version__, sympy.__version__, torch.__version__)"
# Expect roughly: 1.26.4 1.17.1 1.14.0 2.2.2
```

---



## 5. Option B — Create conda env `comp_pde` (recommended)

From the project’s `work/pinn_learning` directory:

```bash
cd work/pinn_learning

conda create -n comp_pde python=3.11 -y
conda activate comp_pde
python -m pip install --upgrade pip
pip install -r requirements-comp_pde.txt && \
python -m ipykernel install --user --name=comp_pde --display-name="comp_pde"
```

If `torch` wheels fail on Apple Silicon, install `torch==2.2.2` using [PyTorch Get Started](https://pytorch.org/get-started/locally/) (pip, macOS), then:

```bash
pip install numpy==1.26.4 matplotlib==3.10.9 scipy==1.17.1 sympy==1.14.0 ipykernel
```

In VS Code, reload the window if `comp_pde` does not appear under **Select Kernel → Python Environments**.

---



## 6. Option C — venv named `comp_pde` workflow

Python **3.11** is required; PyTorch wheels are often missing on **3.14-only** systems.

```bash
cd work/pinn_learning

python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-comp_pde.txt
python -m ipykernel install --user --name=comp_pde --display-name="comp_pde"
```

If an old `.venv` was created with Python 3.13, remove it and recreate with 3.11:

```bash
rm -rf .venv && python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-comp_pde.txt
python -m ipykernel install --user --name=comp_pde --display-name="comp_pde"
```

---



## 7. Checklist for `01_ode_pinn.ipynb`

1. Kernel is `comp_pde`.
2. First code cell prints `device: cpu` (or `mps` / `cuda`).
3. Training finishes and the plot shows PINN vs. e^{-x}.
4. Common issues:
  - `No module named 'torch'` (or scipy/sympy) — install into the kernel’s Python (§5 Option B or §6 Option C).
  - **`requirements-comp_pde.txtpython`** or similar — two commands were pasted on one line; run `pip install -r requirements-comp_pde.txt` and `python -m ipykernel install …` on **separate lines** (or join with `&&`).
  - **NumPy 2.x warnings** — use `requirements-comp_pde.txt` (`numpy==1.26.4`).
  - **Kernel missing** — rerun `python -m ipykernel install --user --name=comp_pde --display-name="comp_pde"`.

```bash
python -c "import numpy, scipy, sympy, torch, matplotlib.pyplot as plt; print('ok')"
```

---



## 8. Refreshing pinned versions

After you upgrade `comp_pde`, update the pin file:

```bash
conda activate comp_pde
python --version
pip freeze | grep -E '^(numpy|matplotlib|torch|scipy|sympy)=='
```

Edit `requirements-comp_pde.txt` accordingly. For loose bounds only, see `requirements.txt`.

---



## 9. Environment-related files


| File                        | Purpose                               |
| --------------------------- | ------------------------------------- |
| `requirements.txt`          | Loose dependencies (Python 3.10–3.13) |
| `requirements-comp_pde.txt` | Pinned stack for env `comp_pde`       |
| `ENV_SETUP.md`              | This guide                            |
| `pinn_utils.py`             | Shared helpers                        |


---



## 10. Student machines

### Editor (required recommendation)

Use **Visual Studio Code** only — see **Quick start** above and **§1**. Extensions: **Python**, **Jupyter** (Microsoft).

### Environment

- **With conda (recommended):** **§5 Option B** — env name `comp_pde`, file `requirements-comp_pde.txt` (includes NumPy, PyTorch, **SciPy**, **SymPy**, Matplotlib, ipykernel).
- **Without conda:** Install Python **3.11**, then **§6 Option C** (venv + the **same** `requirements-comp_pde.txt`). Conda is **not** mandatory for this course.
- **No local setup (backup only):** Google Colab can run short demos, but versions may differ; prefer local VS Code + `comp_pde` for the labs.

### Platform notes

- On Windows/Linux, if `pip install torch` fails, use the pip command from [pytorch.org](https://pytorch.org/get-started/locally/) for your OS, then install the rest from `requirements-comp_pde.txt`.
- Optional: `torchvision==0.17.2` (comment in the pin file).

### Course notebooks

Under `work/pinn_learning/`: start with `01_ode_pinn.ipynb`, then `02_poisson_1d_pinn.ipynb`, etc. Slides point to this file: `work/pinn_learning/ENV_SETUP.md`.