"""
Image geometry from ISMRMRD headers.

ISMRMRD stores, per acquisition or image, the FOV-center `position` and unit
vectors `read_dir`, `phase_dir`, `slice_dir` in the patient coordinate system
(DICOM LPS, mm). `Geometry` bundles these with the field of view and matrix
size and provides the voxel-index -> patient-mm affine. This replaces the
MATLAB `calc_coords_RAW`/`calc_coords_MPME`/`affine_from_meshgrid` helpers.

Index convention: tensors are stored as ``(..., z, y, x)``. Affines act on
column vectors ``(i_x, i_y, i_z, 1)`` (x fastest), like NIfTI. The FOV
center is placed at index ``N // 2`` along each axis, the `fftshift` origin.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Sequence, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Geometry:
    """
    Geometry of a 3D image grid in the patient coordinate system (LPS, mm).

    Attributes
    ----------
    position : tuple of 3 float
        Patient-coordinate position (mm) of the FOV center.
    read_dir, phase_dir, slice_dir : tuple of 3 float
        Unit vectors of the x (readout), y (phase) and z (slice) axes.
    fov : tuple of 3 float
        Field of view (x, y, z) in mm. For 2D slices, z is the slice
        thickness.
    matrix : tuple of 3 int
        Matrix size (x, y, z).
    """
    position: tuple[float, float, float]
    read_dir: tuple[float, float, float]
    phase_dir: tuple[float, float, float]
    slice_dir: tuple[float, float, float]
    fov: tuple[float, float, float]
    matrix: tuple[int, int, int]

    @property
    def resolution(self) -> tuple[float, float, float]:
        """Voxel size (x, y, z) in mm."""
        return tuple(f / m for f, m in zip(self.fov, self.matrix))  # type: ignore

    @property
    def affine(self) -> torch.Tensor:
        """4x4 float64 affine mapping ``(i_x, i_y, i_z, 1)`` to LPS mm."""
        R = torch.tensor([self.read_dir, self.phase_dir, self.slice_dir], dtype=torch.float64).T
        A = torch.eye(4, dtype=torch.float64)
        A[:3, :3] = R * torch.tensor(self.resolution, dtype=torch.float64)
        center = torch.tensor([m // 2 for m in self.matrix], dtype=torch.float64)
        A[:3, 3] = torch.tensor(self.position, dtype=torch.float64) - A[:3, :3] @ center
        return A

    def with_matrix(self, matrix: Sequence[int]) -> "Geometry":
        """Same FOV on a different matrix (e.g. after zero-filled interpolation)."""
        return replace(self, matrix=tuple(int(m) for m in matrix))  # type: ignore

    def with_fov(self, fov: Sequence[float], matrix: Optional[Sequence[int]]=None) -> "Geometry":
        """Same center and orientation with a new FOV (e.g. after cropping)."""
        m = self.matrix if matrix is None else tuple(int(v) for v in matrix)
        return replace(self, fov=tuple(float(f) for f in fov), matrix=m)  # type: ignore

    @classmethod
    def from_header(cls, hdr, fov: Sequence[float], matrix: Sequence[int]) -> "Geometry":
        """
        Build from an `ismrmrd.AcquisitionHeader` or `ismrmrd.ImageHeader`.

        Parameters
        ----------
        hdr : ismrmrd.AcquisitionHeader, ismrmrd.ImageHeader, or numpy record
            Any object with `position`, `read_dir`, `phase_dir`, `slice_dir`
            attributes or fields.
        fov : sequence of 3 float
            Field of view (x, y, z) in mm.
        matrix : sequence of 3 int
            Matrix size (x, y, z).
        """
        def vec(name: str) -> tuple[float, float, float]:
            v = hdr[name] if isinstance(hdr, np.void) else getattr(hdr, name)
            return tuple(float(a) for a in v)  # type: ignore
        return cls(
            position=vec("position"), read_dir=vec("read_dir"),
            phase_dir=vec("phase_dir"), slice_dir=vec("slice_dir"),
            fov=tuple(float(f) for f in fov), matrix=tuple(int(m) for m in matrix),  # type: ignore
        )


def voxel_coordinates(geom: Geometry, dtype: torch.dtype=torch.float64) -> torch.Tensor:
    """
    Patient-coordinate (LPS, mm) position of every voxel center.

    Equivalent to the MATLAB `PosPCS` meshgrid.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(z, y, x, 3)``.
    """
    nx, ny, nz = geom.matrix
    iz, iy, ix = torch.meshgrid(
        torch.arange(nz, dtype=torch.float64),
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij")
    idx = torch.stack([ix, iy, iz, torch.ones_like(ix)], dim=-1)
    return (idx @ geom.affine.T)[..., :3].to(dtype)


def lps_to_ras(affine: torch.Tensor) -> torch.Tensor:
    """Convert an LPS affine (ISMRMRD/DICOM) to RAS (NIfTI)."""
    flip = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=affine.dtype))
    return flip @ affine


def resample(
        image: torch.Tensor,
        source: Geometry,
        target: Geometry,
        mode: str="bilinear",
        padding_mode: str="zeros",
        ) -> torch.Tensor:
    """
    Resample an image from one grid onto another.

    Port of MATLAB `coregister_images(..., 'fast_linear')`: no registration is
    estimated, the scanner geometry is trusted. Complex images are resampled
    as real and imaginary parts.

    Parameters
    ----------
    image : torch.Tensor
        Tensor of shape ``(..., z, y, x)`` on the `source` grid.
    source, target : Geometry
        Source and target grids.
    mode : str, optional
        'bilinear' (trilinear in 3D) or 'nearest'. Defaults to 'bilinear'.
    padding_mode : str, optional
        Passed to `torch.nn.functional.grid_sample`. Defaults to 'zeros'.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(..., z', y', x')`` on the `target` grid.
    """
    if torch.is_complex(image):
        return torch.complex(
            resample(image.real, source, target, mode, padding_mode),
            resample(image.imag, source, target, mode, padding_mode))

    lead = image.shape[:-3]
    vol = image.reshape(1, -1, *image.shape[-3:])  # (1, C, z, y, x)

    # target voxel -> source fractional index (x, y, z)
    M = torch.linalg.solve(source.affine, target.affine)
    nx, ny, nz = target.matrix
    iz, iy, ix = torch.meshgrid(
        torch.arange(nz, dtype=torch.float64),
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij")
    idx = torch.stack([ix, iy, iz, torch.ones_like(ix)], dim=-1) @ M.T
    # normalize to [-1, 1] with align_corners=True
    size = torch.tensor(source.matrix, dtype=torch.float64)
    denom = torch.clamp(size - 1, min=1)
    grid = (2 * idx[..., :3] / denom - 1).to(dtype=image.dtype, device=image.device)[None]

    out = F.grid_sample(vol, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    out = out.reshape(*lead, nz, ny, nx)

    # grid_sample maps every point onto a singleton axis; keep only target
    # voxels within half a voxel of it (e.g. a 2D slice resampled into 3D)
    singleton = size == 1
    if singleton.any() and padding_mode == "zeros":
        inside = (idx[..., :3].abs() <= 0.5)[..., singleton].all(dim=-1)
        out = out * inside.to(dtype=out.dtype, device=out.device)
    return out


def fov_intersection_mask(image_geom: Geometry, other: Geometry, tol: float=1e-3) -> torch.Tensor:
    """
    Boolean ``(z, y, x)`` mask of `image_geom` voxels inside `other`'s FOV.

    Replaces the MATLAB `PosDCSIntersectMasks` helper.
    """
    coords = voxel_coordinates(image_geom)
    hom = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=-1)
    idx = (hom @ torch.linalg.inv(other.affine).T)[..., :3]
    size = torch.tensor(other.matrix, dtype=torch.float64)
    return ((idx >= -0.5 - tol) & (idx <= size - 0.5 + tol)).all(dim=-1)
