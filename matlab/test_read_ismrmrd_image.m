function test_read_ismrmrd_image(h5path, expected_mat)
% TEST_READ_ISMRMRD_IMAGE  Check read_ismrmrd_image against values written by
% tests/test_matlab_reader.py (run via `matlab -batch`; errors fail the test).

E = load(expected_mat);

% real-valued image with non-loop lead dims (pathway, echo, slice)
[d, info] = read_ismrmrd_image(h5path, 'recon');
assert(isequal(size(d), size(E.recon)), 'recon size %s vs %s', mat2str(size(d)), mat2str(size(E.recon)));
assert(max(abs(double(d(:)) - double(E.recon(:)))) == 0, 'recon values differ');
assert(isequal(info.dims, cellstr(E.recon_dims)), 'recon dims differ: %s', strjoin(info.dims, ','));
assert(strcmp(info.meta.units, 'ms'), 'meta units');
assert(isequal(double(info.meta.source_mids(:)'), [373, 374]), 'meta source_mids');
assert(numel(info.slices) == size(E.affines, 3), 'number of slices');
for k = 1:numel(info.slices)
    assert(max(abs(info.slices(k).affine(:) - reshape(E.affines(:, :, k), [], 1))) < 1e-6, ...
           'affine of slice %d differs', k);
end
assert(~isempty(info.xml), 'xml header missing');

% complex coil image
[c, cinfo] = read_ismrmrd_image(h5path, 'coils');
assert(isequal(size(c), size(E.coils)), 'coils size %s vs %s', mat2str(size(c)), mat2str(size(E.coils)));
assert(max(abs(c(:) - E.coils(:))) < 1e-6, 'coil values differ');
assert(isequal(cinfo.dims, cellstr(E.coils_dims)), 'coil dims differ: %s', strjoin(cinfo.dims, ','));

% 2D image without lead dims (nz = 1)
[s1, sinfo] = read_ismrmrd_image(h5path, 'single');
assert(isequal(size(s1), size(E.single)), 'single size %s', mat2str(size(s1)));
assert(isequal(double(s1), double(E.single)), 'single values differ');
assert(isequal(sinfo.dims, {'x', 'y', 'z'}), 'single dims: %s', strjoin(sinfo.dims, ','));

% group without MRITorch metadata -> raw (x, y, z, coil, image) layout
[p1, pinfo] = read_ismrmrd_image(h5path, 'plain');
assert(isequal(double(p1), double(E.single)), 'plain values differ');
assert(isequal(pinfo.dims, {'x', 'y', 'z', 'coil', 'image'}), 'plain dims: %s', strjoin(pinfo.dims, ','));

disp('test_read_ismrmrd_image: OK');
end
