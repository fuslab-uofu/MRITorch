"""
Siemens twix (.dat) -> ISMRMRD conversion via the `siemens_to_ismrmrd` CLI.

The converter's default stylesheet copies every ``sWipMemBlock.alFree[i]`` and
``sWipMemBlock.adFree[i]`` (0-based) into `userParameters`, so WIP parameters
(e.g. MPME pathway codes, golden-angle step, AFI TR) are available from
`mritorch.io.header.user_parameters` without a custom stylesheet. MATLAB
mapVBVD's ``alFree{n}`` corresponds to ``alFree[n-1]``. Check this against a
real file: XProtocol drops unset entries, which can shift the numbering.

The converter is an optional external C++ tool; it is not installed with
MRITorch. Install it from the `ismrmrd` conda channel with the repository's
``environment.yml`` (``conda create -n siemens2ismrmrd --no-default-packages``,
then ``conda env update -n siemens2ismrmrd -f environment.yml``; see the
README), and set ``MRITORCH_SIEMENS_TO_ISMRMRD`` to the converter path if it is
not on ``PATH``.
"""
from __future__ import annotations
from typing import Optional, Sequence
from pathlib import Path

import os
import shutil
import subprocess


def find_converter() -> str:
    """Path to `siemens_to_ismrmrd`, from the environment or ``PATH``."""
    exe = os.environ.get("MRITORCH_SIEMENS_TO_ISMRMRD") or shutil.which("siemens_to_ismrmrd")
    if not exe:
        raise FileNotFoundError(
            "siemens_to_ismrmrd not found. It is an optional external tool: install it "
            "from the MRITorch repository with `conda create -n siemens2ismrmrd "
            "--no-default-packages` and then `conda env update -n siemens2ismrmrd "
            "-f environment.yml` (see the README), then set "
            "MRITORCH_SIEMENS_TO_ISMRMRD to its path or put it on PATH.")
    return exe


def siemens_to_ismrmrd(
        dat_file: str | Path,
        output: Optional[str | Path]=None,
        measurement: int=-1,
        group: str="dataset",
        user_stylesheet: Optional[str | Path]=None,
        user_map: Optional[str | Path]=None,
        extra_args: Sequence[str]=(),
        noise: bool=False,
        noise_group: str="noise",
        overwrite: bool=False,
        ) -> Path:
    """
    Convert a Siemens .dat file to ISMRMRD HDF5.

    Parameters
    ----------
    dat_file : str or Path
        Siemens raw data file.
    output : str or Path, optional
        Output .h5 path. Defaults to `dat_file` with a .h5 suffix.
    measurement : int, optional
        Measurement number within a multi-RAID file; negative counts from
        the end. Defaults to -1, the imaging measurement.
    group : str, optional
        HDF5 group name. Defaults to 'dataset'.
    user_stylesheet, user_map : str or Path, optional
        Custom parameter stylesheet (.xsl) and map (.xml).
    extra_args : sequence of str, optional
        Additional converter arguments.
    noise : bool, optional
        Also convert measurement 1 of a multi-RAID file (the Siemens noise
        adjustment) into group `noise_group` of the same output file, where
        `mritorch.io.ismrmrd.read_kspace` picks up its noise acquisitions.
        Defaults to False.
    noise_group : str, optional
        HDF5 group for the noise measurement. Defaults to 'noise'.
    overwrite : bool, optional
        Replace an existing output file. Defaults to False: the converter
        appends to existing files, which would silently duplicate data.

    The converter runs with ``HDF5_USE_FILE_LOCKING=FALSE`` in its
    environment (only for the subprocess), since default HDF5 locking hangs
    on NFS mounts without lock support.

    Returns
    -------
    Path
        Path of the written ISMRMRD file.
    """
    dat_file = Path(dat_file)
    output = Path(output) if output is not None else dat_file.with_suffix(".h5")
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass overwrite=True to replace it")
    if noise and noise_group == group:
        raise ValueError("noise_group must differ from group")
    exe = find_converter()
    # convert into a temporary file and rename on success, so a failed run
    # neither leaves a partial output nor destroys an existing one
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        partial.unlink()
    env = {**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE"}
    style = []
    if user_stylesheet is not None:
        style += ["-x", str(user_stylesheet)]
    if user_map is not None:
        style += ["-m", str(user_map)]
    runs = [(measurement, group)] + ([(1, noise_group)] if noise else [])
    try:
        for meas, grp in runs:
            cmd = [exe, "-f", str(dat_file), "-o", str(partial), "-z", str(meas), "-g", grp, *style, *extra_args]
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"siemens_to_ismrmrd failed ({result.returncode}):\n{result.stderr}")
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    os.replace(partial, output)
    return output
