# MRITorch

A physics-informed PyTorch library for quantitative MRI.

## Installation

MRITorch requires Python 3.11 or newer. The core library needs PyTorch, NumPy
and [torchkbnufft](https://torchkbnufft.readthedocs.io) (which brings SciPy):

```bash
pip install mritorch
```

Optional features are installed as extras, e.g. `pip install "mritorch[ismrmrd,nifti]"`:

| Extra     | Enables                                                   | Installs          |
|-----------|-----------------------------------------------------------|-------------------|
| `ismrmrd` | Reading and writing ISMRMRD HDF5 k-space and images (`mritorch.io`) | ismrmrd, h5py |
| `nifti`   | NIfTI export (`mritorch.io.to_nifti`)                    | nibabel           |
| `mat`     | MATLAB `.mat` loading (`mritorch.io.load_mat`)            | scipy, h5py       |
| `all`     | All of the above                                          |                   |

Everything else (FFTs, coil combination, ESPIRiT/SENSE, Cartesian and radial
reconstruction, EPG simulation) works on plain tensors with no extras.

**Intel Macs:** PyTorch 2.2.2 is the last release for macOS on x86_64, and it
does not work with NumPy 2. MRITorch's metadata restricts NumPy to `<2` on
that platform, so pip resolves a working combination automatically. On all
other platforms MRITorch requires PyTorch 2.3 or newer, which works with NumPy 2.

### Radial reconstruction

The radial NUFFT (`mritorch.recon.radial.Nufft2d`) uses torchkbnufft by
default (`backend="kb"`), so it is differentiable with autograd and runs on the
device of its input. Accuracy is set by `numpoints` (kernel width, default 6)
and `oversamp` (grid oversampling, default 2.0); the defaults agree with the
exact `backend="direct"` sums to a relative error of about 1e-5.
`reconstruct_mpme` and `calibration_maps` accept the same `backend`,
`numpoints` and `oversamp` arguments. CPU
parallelism follows `torch.set_num_threads`. On GPU, the NUFFT is tested on
CUDA (`python -m pytest --run-gpu`); the other reconstruction steps are only
tested on CPU so far.

### Siemens raw data (optional)

Converting Siemens twix `.dat` files to ISMRMRD uses the external
[siemens_to_ismrmrd](https://github.com/ismrmrd/siemens_to_ismrmrd) C++ tool,
which pip cannot install. MRITorch only calls it as a subprocess, so it can
live in its own conda environment:

```bash
conda create -n siemens2ismrmrd --no-default-packages --yes
conda env update -n siemens2ismrmrd -f environment.yml
```

Creating the environment empty first keeps out any `create_default_packages`
from your `.condarc`. Those would pull in the newest Python, and the converter's
Boost dependency only has builds for Python 3.11 and older, so
`conda env create -f environment.yml` fails with an unsatisfiable `boost`
requirement when default packages are configured. Use any name in place of
`siemens2ismrmrd`.

Then point MRITorch at the converter. MRITorch looks for it in the
`MRITORCH_SIEMENS_TO_ISMRMRD` environment variable first, then on `PATH`.
Setting the variable is safer than adding the conda environment's `bin` to
`PATH`, which can shadow the Python of your working environment. Find the path
with `conda run -n siemens2ismrmrd which siemens_to_ismrmrd`, then add it to
your shell startup file:

```bash
# bash (~/.bashrc)
export MRITORCH_SIEMENS_TO_ISMRMRD=/path/to/envs/siemens2ismrmrd/bin/siemens_to_ismrmrd
```

```tcsh
# tcsh (~/.tcshrc)
setenv MRITORCH_SIEMENS_TO_ISMRMRD /path/to/envs/siemens2ismrmrd/bin/siemens_to_ismrmrd
```

Check that MRITorch finds it (restart any open Jupyter or VS Code kernels first):

```bash
python -c "import subprocess; from mritorch.io.convert import find_converter; subprocess.run([find_converter(), '--version'], check=True)"
```

## Setting up a general reconstruction environment

These steps give you a [uv](https://docs.astral.sh/uv/) environment with
MRITorch and GPU PyTorch, plus the converter for Siemens twix `.dat` files.
This is my setup for general image reconstruction work in Python.

The environment lives on the node's local disk rather than in your home
directory (which may have a quota) or on network storage (where Python
environments are slow). Each node gets its own copy, and uv installs the
PyTorch CUDA build that matches that node's NVIDIA driver
(`--torch-backend=auto`; CPU-only on nodes without a GPU). The paths below use
`/working/$USER`; substitute your node-local disk.

**1. Point uv at the local disk.** Add to your shell startup file, so the
cache, uv's Python installs and the environment are all on the same disk (uv
then hard-links packages instead of copying them):

```tcsh
# tcsh (~/.tcshrc)
setenv UV_CACHE_DIR /working/$USER/uv/cache
setenv UV_PYTHON_INSTALL_DIR /working/$USER/uv/python
```

```bash
# bash (~/.bashrc)
export UV_CACHE_DIR=/working/$USER/uv/cache
export UV_PYTHON_INSTALL_DIR=/working/$USER/uv/python
```

**2. Create the environment** on each node you use (open a new shell first so
the settings above apply):

```bash
uv venv --python 3.11 /working/$USER/uv/venvs/mri-recon
uv pip install --python /working/$USER/uv/venvs/mri-recon/bin/python --torch-backend=auto \
    "mritorch[all] @ git+https://github.com/fuslab-uofu/MRITorch" matplotlib ipykernel
```

Add any other packages you use to the `uv pip install` line. To update
MRITorch to the latest code later, rerun that line with
`--upgrade-package mritorch --refresh-package mritorch` added, so uv fetches
the newest commit instead of reusing its cached copy. To use the environment:

```bash
source /working/$USER/uv/venvs/mri-recon/bin/activate.csh   # tcsh
source /working/$USER/uv/venvs/mri-recon/bin/activate       # bash
```

For Jupyter and VS Code notebooks, register the environment as a kernel once
(with it activated): `python -m ipykernel install --user --name mri-recon`.
The kernel points at `/working/$USER/...`, the same path on every node, so it
works on any node where you have built the environment.

**3. Install the Siemens converter** in its own conda environment and set
`MRITORCH_SIEMENS_TO_ISMRMRD`, as described in
[Siemens raw data](#siemens-raw-data-optional).

**4. Check the setup** with the environment activated:

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import subprocess; from mritorch.io.convert import find_converter; subprocess.run([find_converter(), '--version'], check=True)"
```

The first should print `True` on a GPU node, and the second the converter and
ISMRMRD versions; an error means `MRITORCH_SIEMENS_TO_ISMRMRD` is unset or
points to the wrong file.

## Development

Development uses [uv](https://docs.astral.sh/uv/). From a clone of the
repository:

```bash
uv sync                       # creates .venv and installs MRITorch in editable mode
uv run pytest                 # or: source .venv/bin/activate, then python -m pytest
uv run pytest --run-gpu       # also run the tests that need a CUDA device
```

`uv sync` installs the versions pinned in `uv.lock`, on the Python version in
`.python-version` (3.11, the oldest supported), so every developer tests
against the same packages. It installs the `dev` dependency group, defined in
`pyproject.toml` and not part of the published package: pytest, scipy and all
optional extras, so no test is skipped for a missing dependency. After changing
dependencies in `pyproject.toml`, run `uv lock` and commit the updated
`uv.lock`; `uv add <package>` and `uv add --dev <package>` do both steps.

If the clone is on network storage, Python environments there are slow. Keep
the environment on a local disk and link it into the clone; `uv sync` follows
the link:

```bash
uv venv --python 3.11 /working/$USER/uv/venvs/mritorch-dev
ln -s /working/$USER/uv/venvs/mritorch-dev .venv
uv sync
```

Without uv, an editable install with pip (version 25.1 or newer, for `--group`)
also works, but does not use the pinned versions in `uv.lock`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e . --group dev
python -m pytest
```
