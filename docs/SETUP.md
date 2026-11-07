# Local Setup

## Portable Repository Checks

Python 3.12 and Git are sufficient for the standard-library publication checks:

```powershell
python tools/check_repo.py
python -m unittest discover -s tests/repository -v
```

These commands neither install dependencies nor contact external services.

## Research Environment

The recorded environment targets Windows x64, Python 3.12 and the pinned CUDA
PyTorch build. `requirements.in` and `requirements.lock` are unchanged from the
reviewed local environment. Follow the index and hash requirements in the lock;
do not replace the pinned CUDA wheel with an unreviewed package.

The existing machine already has the required `.venv`. Use its Python directly;
no activation or PowerShell execution-policy change is needed:

```powershell
.\.venv\Scripts\python.exe scripts/lab.py prepare --with-local-data
.\.venv\Scripts\python.exe scripts/lab.py test
.\.venv\Scripts\python.exe scripts/lab.py verify
.\.venv\Scripts\python.exe scripts/lab.py notebook
```

Creating a new environment and downloading packages/data are separate deliberate
operations, not automatic behavior of these tools. Review
[the dependency record](dependencies.md) before provisioning another environment.
It records a dated advisory check, not a guarantee about future advisories.

## Data And Historical Fixtures

The repository does not ship weights, dataset questions, raw model responses or
the original executed notebooks. The full 140-test historical suite therefore
requires the existing local audit pack, cached model/tokenizer and published
edit arrays. Fresh clones can run portable checks and the curated results
overview; missing historical fixtures are reported explicitly.

The archived scripts retain the explicit cached-data download helpers, but they
are not called by repository checks or the launcher. Any new download or inference
requires a separate deliberate action.

## Notebooks

Jupyter opens notebook 00, a display-only overview of the curated aggregate JSON.
Full-analysis notebooks use the local workbench. Do not share a Jupyter login
URL or disable its default authentication. Keep the server bound to loopback.

Notebooks 01/02 contain GPU demonstrations behind an explicit opt-in. Notebook
03's optional activation pass defaults off. Notebooks 04-07 keep their run flags
off and need local saved results for their complete views. Stop unused GPU
kernels before opening another model notebook.
