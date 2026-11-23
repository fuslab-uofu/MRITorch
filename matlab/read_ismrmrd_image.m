function [data, info] = read_ismrmrd_image(path, group, dataset)
% READ_ISMRMRD_IMAGE  Read an ISMRMRD image group (as written by MRITorch) into MATLAB order.
%
%   [data, info] = read_ismrmrd_image(path, group)
%   [data, info] = read_ismrmrd_image(path, group, dataset)
%
% Inputs
%   path    - ISMRMRD HDF5 file
%   group   - image group name, e.g. 'recon' or 't1'
%   dataset - top-level group (default 'dataset')
%
% Outputs
%   data - array in MATLAB order: (x, y, z, [coil], lead dims...). For a group
%          written by mritorch.io.write_image, the lead dims are the Python
%          lead dims in their original order, e.g. a Python Image with dims
%          (pathway, echo, z, y, x) becomes data(x, y, z, pathway, echo).
%          Groups without MRITorch metadata are returned as (x, y, z, coil, image).
%   info - struct with fields
%          dims     - cell array naming each axis of data
%          meta     - Image.meta stored by MRITorch (struct; empty if none)
%          xml      - the file's ISMRMRD XML header (char; '' if none)
%          slices   - struct array per slice counter: slice, position,
%                     read_dir, phase_dir, slice_dir, fov, matrix, affine.
%                     affine maps 0-based voxel indices (ix, iy, iz, 1) to
%                     LPS mm, identical to mritorch.geometry.Geometry.affine.
%                     For 1-based MATLAB subscripts use affine * [i-1; j-1; k-1; 1].
%
% IMPORTANT (NFS mounts without lock support, e.g. /ceph): start MATLAB with
% HDF5_USE_FILE_LOCKING=FALSE already set in the launching shell, e.g.
%     HDF5_USE_FILE_LOCKING=FALSE matlab        (bash)
%     setenv HDF5_USE_FILE_LOCKING FALSE        (tcsh, before starting MATLAB)
% Otherwise h5read hangs. Calling setenv inside MATLAB is NOT enough (tested:
% a MATLAB started without the variable hung on a /ceph file even with the
% setenv below); it is kept only as a harmless default. The function warns if
% the variable was not set when MATLAB started.

if nargin < 3 || isempty(dataset)
    dataset = 'dataset';
end
if ~strcmpi(getenv('HDF5_USE_FILE_LOCKING'), 'FALSE')
    warning('read_ismrmrd_image:locking', ['HDF5_USE_FILE_LOCKING is not FALSE; reads on NFS (e.g. /ceph) ' ...
            'may hang. Restart MATLAB with HDF5_USE_FILE_LOCKING=FALSE set in the shell.']);
    setenv('HDF5_USE_FILE_LOCKING', 'FALSE');
end

base = ['/' dataset '/' group];
hdr = h5read(path, [base '/header']);
raw = h5read(path, [base '/data']);
if isstruct(raw)
    raw = complex(raw.(first_field(raw, {'real', 'r'})), raw.(first_field(raw, {'imag', 'i'})));
end
% h5read reverses the HDF5 (n, coil, z, y, x) layout to (x, y, z, coil, n)

attrs = {};
try
    attrs = h5read(path, [base '/attributes']);
catch
end
xml = '';
try
    x = h5read(path, ['/' dataset '/xml']);
    if iscell(x), x = x{1}; end
    xml = char(x);
catch
end

% order images by image_index
[~, order] = sort(double(hdr.image_index(:)));
raw = raw(:, :, :, :, order);

stored = parse_meta(attrs);
nx = size(raw, 1); ny = size(raw, 2); nz = size(raw, 3); nch = size(raw, 4);
if isempty(stored)
    data = raw;
    info.dims = {'x', 'y', 'z', 'coil', 'image'};
    info.meta = struct();
else
    pydims = cellstr(stored.dims(:)');
    pyshape = double(stored.shape(:)');
    has_coil = any(strcmp(pydims, 'coil'));
    nlead = numel(pydims) - 3 - has_coil;
    lead = pyshape(1:nlead);
    % Python flattens lead dims in C order (last lead fastest), which is
    % MATLAB column-major order over the reversed lead dims
    data = reshape(raw, [nx, ny, nz, nch, fliplr(lead), 1]);
    data = permute(data, [1:4, (4 + nlead):-1:5, (5 + nlead)]);
    leadnames = pydims(1:nlead);
    if has_coil
        info.dims = [{'x', 'y', 'z', 'coil'}, leadnames];
    else
        sz = size(data, 1:max(5, ndims(data)));
        data = reshape(data, [sz(1:3), sz(5:end), 1]);
        info.dims = [{'x', 'y', 'z'}, leadnames];
    end
    info.meta = stored.meta;
end
info.xml = xml;
info.slices = slice_geometry(hdr);
end


function name = first_field(s, candidates)
f = fieldnames(s);
name = '';
for k = 1:numel(candidates)
    if any(strcmp(f, candidates{k}))
        name = candidates{k};
        return
    end
end
error('read_ismrmrd_image:complex', 'unexpected complex field names: %s', strjoin(f', ', '));
end


function stored = parse_meta(attrs)
stored = [];
if isempty(attrs)
    return
end
if iscell(attrs), a = attrs{1}; else, a = attrs(1, :); end
a = char(a);
tok = regexp(a, '<meta><name>mritorch</name><value>(.*?)</value></meta>', 'tokens', 'once');
if isempty(tok)
    return
end
js = tok{1};
js = strrep(js, '&quot;', '"');
js = strrep(js, '&apos;', '''');
js = strrep(js, '&lt;', '<');
js = strrep(js, '&gt;', '>');
js = strrep(js, '&amp;', '&');
stored = jsondecode(js);
end


function slices = slice_geometry(hdr)
s_all = double(hdr.slice(:));
[u, first] = unique(s_all, 'first');
slices = struct('slice', {}, 'position', {}, 'read_dir', {}, 'phase_dir', {}, 'slice_dir', {}, ...
                'fov', {}, 'matrix', {}, 'affine', {});
for k = 1:numel(u)
    j = first(k);
    g.slice = u(k);
    g.position = col(hdr.position, j);
    g.read_dir = col(hdr.read_dir, j);
    g.phase_dir = col(hdr.phase_dir, j);
    g.slice_dir = col(hdr.slice_dir, j);
    g.fov = col(hdr.field_of_view, j);
    g.matrix = col(hdr.matrix_size, j);
    res = g.fov ./ g.matrix;
    A = eye(4);
    A(1:3, 1:3) = [g.read_dir, g.phase_dir, g.slice_dir] .* res';
    center = floor(g.matrix / 2);
    A(1:3, 4) = g.position - A(1:3, 1:3) * center;
    g.affine = A;
    slices(k) = g; %#ok<AGROW>
end
end


function v = col(field, j)
% header array fields come back as (3, n)
v = double(field(:, j));
end
