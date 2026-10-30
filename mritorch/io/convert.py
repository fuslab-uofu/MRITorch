"""
Siemens twix (.dat) -> ISMRMRD conversion via the `siemens_to_ismrmrd` CLI.

The converter's default stylesheet copies every ``sWipMemBlock.alFree[i]`` and
``sWipMemBlock.adFree[i]`` (0-based) into `userParameters`, so WIP parameters
(e.g. MPME pathway codes, golden-angle step, AFI TR) are available from
`mritorch.io.header.user_parameters` without a custom stylesheet. MATLAB
mapVBVD's ``alFree{n}`` corresponds to ``alFree[n-1]``. Check this against a
real file: XProtocol drops unset entries, which can shift the numbering.

Set ``MRITORCH_SIEMENS_TO_ISMRMRD`` to the converter path if it is not on
``PATH``.
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
            "siemens_to_ismrmrd not found. Build it (Resources/Code/siemens_to_ismrmrd) "
            "and put it on PATH or set MRITORCH_SIEMENS_TO_ISMRMRD.")
    return exe


def siemens_to_ismrmrd(
        dat_file: str | Path,
        output: Optional[str | Path]=None,
        measurement: int=-1,
        group: str="dataset",
        user_stylesheet: Optional[str | Path]=None,
        user_map: Optional[str | Path]=None,
        extra_args: Sequence[str]=(),
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

    Returns
    -------
    Path
        Path of the written ISMRMRD file.
    """
    dat_file = Path(dat_file)
    output = Path(output) if output is not None else dat_file.with_suffix(".h5")
    cmd = [find_converter(), "-f", str(dat_file), "-o", str(output),
           "-z", str(measurement), "-g", group]
    if user_stylesheet is not None:
        cmd += ["--user-stylesheet", str(user_stylesheet)]
    if user_map is not None:
        cmd += ["--user-map", str(user_map)]
    cmd += list(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"siemens_to_ismrmrd failed ({result.returncode}):\n{result.stderr}")
    return output
