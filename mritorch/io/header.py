"""
Access to the ISMRMRD XML header.

The parsed header is an `ismrmrd.xsd.ismrmrdHeader` dataclass. These helpers
pull out the values the reconstruction and fitting code needs. ISMRMRD stores
TR, TE and TI in ms and flip angles in degrees.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union


def parse_header(xml: Union[str, bytes]):
    """Parse ISMRMRD XML into an `ismrmrd.xsd.ismrmrdHeader`."""
    import ismrmrd.xsd
    return ismrmrd.xsd.CreateFromDocument(xml)


def header_to_xml(header) -> str:
    """Serialize an `ismrmrd.xsd.ismrmrdHeader` to XML."""
    import ismrmrd.xsd
    return ismrmrd.xsd.ToXML(header)


@dataclass(frozen=True)
class EncodingInfo:
    """
    Encoded and reconstructed spaces of one ISMRMRD encoding.

    Matrix sizes are (x, y, z); FOVs are (x, y, z) in mm. Centers are the
    k-space center indices from `encodingLimits` (None if not given).
    """
    encoded_matrix: tuple[int, int, int]
    encoded_fov: tuple[float, float, float]
    recon_matrix: tuple[int, int, int]
    recon_fov: tuple[float, float, float]
    center_ky: Union[int, None]
    center_kz: Union[int, None]
    trajectory: str


def encoding_info(header, encoding: int=0) -> EncodingInfo:
    """Matrix sizes, FOVs and k-space centers of `header.encoding[encoding]`."""
    enc = header.encoding[encoding]

    def mat(space) -> tuple[int, int, int]:
        m = space.matrixSize
        return (int(m.x), int(m.y), int(m.z))

    def fov(space) -> tuple[float, float, float]:
        f = space.fieldOfView_mm
        return (float(f.x), float(f.y), float(f.z))

    def center(limit) -> Union[int, None]:
        return None if limit is None else int(limit.center)

    lim = enc.encodingLimits
    traj = enc.trajectory
    return EncodingInfo(
        encoded_matrix=mat(enc.encodedSpace), encoded_fov=fov(enc.encodedSpace),
        recon_matrix=mat(enc.reconSpace), recon_fov=fov(enc.reconSpace),
        center_ky=center(lim.kspace_encoding_step_1) if lim is not None else None,
        center_kz=center(lim.kspace_encoding_step_2) if lim is not None else None,
        trajectory=getattr(traj, "value", str(traj)),
    )


def make_header(
        encoded_matrix: tuple[int, int, int],
        encoded_fov: tuple[float, float, float],
        recon_matrix: Optional[tuple[int, int, int]]=None,
        recon_fov: Optional[tuple[float, float, float]]=None,
        center_ky: Optional[int]=None,
        center_kz: Optional[int]=None,
        trajectory: str="cartesian",
        TE: Sequence[float]=(),
        TR: Sequence[float]=(),
        TI: Sequence[float]=(),
        flip_angle: Sequence[float]=(),
        user_long: Optional[dict[str, int]]=None,
        user_double: Optional[dict[str, float]]=None,
        ):
    """
    Build a single-encoding `ismrmrd.xsd.ismrmrdHeader`.

    Used for synthetic data and for converters that write ISMRMRD from other
    formats. Matrix sizes and FOVs are (x, y, z); times are in ms.
    """
    import ismrmrd.xsd as x

    def space(m, f):
        return x.encodingSpaceType(
            matrixSize=x.matrixSizeType(x=m[0], y=m[1], z=m[2]),
            fieldOfView_mm=x.fieldOfViewMm(x=f[0], y=f[1], z=f[2]))

    def limit(n, c):
        return x.limitType(minimum=0, maximum=n - 1, center=n // 2 if c is None else c)

    recon_matrix = encoded_matrix if recon_matrix is None else recon_matrix
    recon_fov = encoded_fov if recon_fov is None else recon_fov
    seq = x.sequenceParametersType(TE=list(TE), TR=list(TR), TI=list(TI),
                                   flipAngle_deg=list(flip_angle))
    user = x.userParametersType(
        userParameterLong=[x.userParameterLongType(name=k, value=v) for k, v in (user_long or {}).items()],
        userParameterDouble=[x.userParameterDoubleType(name=k, value=v) for k, v in (user_double or {}).items()])
    return x.ismrmrdHeader(
        encoding=[x.encodingType(
            encodedSpace=space(encoded_matrix, encoded_fov),
            reconSpace=space(recon_matrix, recon_fov),
            encodingLimits=x.encodingLimitsType(
                kspace_encoding_step_1=limit(encoded_matrix[1], center_ky),
                kspace_encoding_step_2=limit(encoded_matrix[2], center_kz)),
            trajectory=x.trajectoryType(trajectory))],
        sequenceParameters=seq, userParameters=user)


def sequence_parameters(header) -> dict[str, list[float]]:
    """
    TR, TE, TI (ms), flip angles (deg) and echo spacing from the header.

    Missing entries are returned as empty lists.
    """
    sp = header.sequenceParameters
    keys = {"TR": "TR", "TE": "TE", "TI": "TI", "flip_angle": "flipAngle_deg",
            "echo_spacing": "echo_spacing"}
    if sp is None:
        return {k: [] for k in keys}
    return {k: [float(v) for v in (getattr(sp, a, None) or [])] for k, a in keys.items()}


def user_parameters(header) -> dict[str, Any]:
    """
    All `userParameters` as a flat ``{name: value}`` dict.

    Siemens WIP parameters mapped by the converter stylesheet (e.g.
    ``sWipMemBlock.alFree[2]``) appear here under their stylesheet names.
    """
    up = header.userParameters
    if up is None:
        return {}
    out: dict[str, Any] = {}
    for group in ("userParameterLong", "userParameterDouble", "userParameterString",
                  "userParameterBase64"):
        for p in getattr(up, group, None) or []:
            out[p.name] = p.value
    return out
