# MRITorch

A physics-informed PyTorch library for quantitative MRI.

## Installation

The conda environment installs MRITorch with all optional dependencies plus
the [siemens_to_ismrmrd](https://github.com/ismrmrd/siemens_to_ismrmrd)
converter, a C++ tool that pip cannot install:

```bash
conda env create -f environment.yml
conda activate mritorch
```

To install into an existing environment instead:

```bash
pip install -e ".[io,nufft,dev]"
conda install -c conda-forge -c ismrmrd siemens_to_ismrmrd "ismrmrd::ismrmrd=1.14.*"
```

If the converter is installed elsewhere, set `MRITORCH_SIEMENS_TO_ISMRMRD` to
its path.
