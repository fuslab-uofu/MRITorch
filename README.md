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
that platform, so pip resolves a working combination automatically.

### Radial reconstruction

The radial NUFFT (`mritorch.recon.radial.Nufft2d`) uses torchkbnufft by
default (`backend="kb"`), so it is differentiable with autograd and runs on the
device of its input. Accuracy is set by `numpoints` (kernel width, default 6)
and `oversamp` (grid oversampling, default 2.0); the defaults agree with the
exact `backend="direct"` sums to a relative error of about 1e-5.
`reconstruct_mpme` and `calibration_maps` accept the same `backend`,
`numpoints` and `oversamp` arguments. CPU
parallelism follows `torch.set_num_threads`. GPU use has not been tested yet.

### Siemens raw data (optional)

Converting Siemens twix `.dat` files to ISMRMRD uses the external
[siemens_to_ismrmrd](https://github.com/ismrmrd/siemens_to_ismrmrd) C++ tool,
which pip cannot install. MRITorch only calls it as a subprocess, so it can
live in its own conda environment:

```bash
conda env create -f environment.yml             # creates an env named siemens2ismrmrd
conda env create -f environment.yml -n <name>   # or choose your own name
```

Then put the converter on your `PATH`, or point MRITorch at it:

```bash
export MRITORCH_SIEMENS_TO_ISMRMRD="$(conda run -n siemens2ismrmrd which siemens_to_ismrmrd)"
```

## Development

Development uses a virtual environment with an editable install. With
[uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[all]" --group dev
python -m pytest
```

With plain pip (version 25.1 or newer, for `--group`):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[all]" --group dev
python -m pytest
```

The `dev` dependency group (pytest and scipy, used by the tests) is defined in
`pyproject.toml` and is not part of the published package. Tests that need an
optional dependency are skipped when it is not installed.
