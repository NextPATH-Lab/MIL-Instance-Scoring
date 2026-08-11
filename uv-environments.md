# Using `uv` — Quick Reference

`uv` can operate in three main modes depending on what environment you want it to touch. This guide covers the three most common cases.

---

## 1. Installing into an existing conda env

`uv`'s pip-compatible commands (`uv pip ...`) will detect and use an **activated** conda env automatically, via the `CONDA_PREFIX` environment variable.

```bash
conda activate myenv
uv pip install -r pyproject.toml
# or, for an editable install of the current project:
uv pip install -e .
```

If you don't want to activate the env (e.g. in a script), set `CONDA_PREFIX` manually:

```bash
export CONDA_PREFIX=/path/to/miniconda3/envs/myenv
uv pip install -r pyproject.toml
```

**Notes / gotchas:**
- Use `uv pip install`, **not** `uv sync`. The `uv sync` project command ignores `CONDA_PREFIX` and always targets a `.venv` — it doesn't know about conda envs.
- A useful pattern: let conda manage system-level / compiled dependencies (CUDA toolkit, compilers, Qt, etc.), and let `uv pip install` handle the pure-Python side fast.

---

## 2. Standard `.venv` via `uv sync`

This is `uv`'s native, project-based workflow — one `pyproject.toml` (+ `uv.lock`) → one `.venv`.

```bash
uv sync
```

This creates/updates `.venv/` in your project directory to match `pyproject.toml` and `uv.lock`.

**Activating it** works exactly like a normal venv:

| Shell | Command |
|---|---|
| bash/zsh | `source .venv/bin/activate` |
| fish | `source .venv/bin/activate.fish` |
| PowerShell | `.venv\Scripts\Activate.ps1` |
| cmd.exe | `.venv\Scripts\activate.bat` |

Deactivate with `deactivate`, same as usual.

**What's actually in `.venv`:**
- `.venv/bin/python` is a **symlink** to a Python interpreter `uv` manages centrally (downloaded/cached once, shared across projects).
- Installed packages are **hardlinked** from `uv`'s global cache into `site-packages` when possible (saves disk space), falling back to a full copy if cache and `.venv` are on different filesystems (this is the hardlink warning you may see).

Manually installing extra packages into this env (without touching `pyproject.toml`):

```bash
uv pip install some-package --python .venv/bin/python
```

(Prefer `uv add some-package` if you want it tracked in `pyproject.toml`/lockfile properly.)

---

## 3. Running scripts with `uv run`

`uv run` executes a command inside the project's env **without requiring activation** — it resolves/creates `.venv` on the fly if needed, then runs your command inside it.

```bash
uv run python train.py
uv run pytest
uv run jupyter lab
```

This is the go-to for:
- One-off commands or CI steps where activating a shell is unnecessary overhead.
- Making sure you're always running against the exact locked environment, even if your shell has some other Python active.

**Running a script with its own inline dependencies** (no project needed), using [PEP 723](https://peps.python.org/pep-0723/) metadata:

```bash
uv run --with numpy --with pandas script.py
```

or embed dependencies directly in the script header:

```python
# /// script
# dependencies = ["numpy", "pandas"]
# ///
```
```bash
uv run script.py
```

`uv` will spin up an ephemeral, cached env just for that script's dependencies.

---

## Quick decision guide

| Situation | Command |
|---|---|
| I have a conda env, want to add packages fast | `conda activate env` → `uv pip install ...` |
| I want a project-managed `.venv` from `pyproject.toml` | `uv sync`, then activate normally |
| I just want to run one command against the project env | `uv run <command>` |
| I want to run a standalone script with its own deps, no project | `uv run --with <pkg> script.py` or inline PEP 723 header |