"""
Input/output. ISMRMRD HDF5 is the standard format for k-space and images.

Submodules import their optional dependencies (ismrmrd, h5py, nibabel, scipy)
lazily, so `import mritorch.io` works without them.
"""
from mritorch.io.ismrmrd import read_kspace, write_kspace, read_image, write_image
from mritorch.io.header import (parse_header, header_to_xml, encoding_info, make_header,
                                sequence_parameters, user_parameters)
from mritorch.io.convert import siemens_to_ismrmrd
from mritorch.io.export import to_nifti, load_mat

__all__ = [
    "read_kspace", "write_kspace", "read_image", "write_image",
    "parse_header", "header_to_xml", "encoding_info", "make_header", "sequence_parameters", "user_parameters",
    "siemens_to_ismrmrd", "to_nifti", "load_mat",
]
