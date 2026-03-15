clear; close all; clc;

script_dir = fileparts(mfilename('fullpath'));
cfg = nanorod_config();

c0 = 299792458;
eps0 = 8.854187817e-12;

rod_diameter_nm = to_numeric(cfg.nanorod.diameter_nm);
rod_length_nm = to_numeric(cfg.nanorod.length_nm);
mesh_n = to_numeric(cfg.nanorod.mesh_n);

pmesh = trirod(rod_diameter_nm, rod_length_nm, mesh_n, 'triangles');

rotate_deg = to_numeric(cfg.nanorod.rotate_deg);
rotate_axis = to_row_vector(cfg.nanorod.rotate_axis, 3);
if rotate_deg ~= 0
    pmesh = rot(pmesh, rotate_deg, rotate_axis);
end

if cfg.nanorod.align_left_edge_x0
    min_x = min(pmesh.pos(:, 1));
    pmesh = shift(pmesh, [-min_x, 0, 0]);
end
if cfg.nanorod.align_top_z0
    max_z = max(pmesh.pos(:, 3));
    pmesh = shift(pmesh, [0, 0, -max_z]);
end

x_center = 0.5 * (min(pmesh.pos(:, 1)) + max(pmesh.pos(:, 1)));
y_center = 0.5 * (min(pmesh.pos(:, 2)) + max(pmesh.pos(:, 2)));
z_top = max(pmesh.pos(:, 3));

position_mode = char(cfg.dipole.position_mode);
if strcmpi(position_mode, 'manual')
    dipole_position_nm = [
        to_numeric(cfg.dipole.position_nm.x),
        to_numeric(cfg.dipole.position_nm.y),
        to_numeric(cfg.dipole.position_nm.z)
    ];
else
    dipole_offset_nm = to_numeric(cfg.dipole.offset_nm);
    dipole_position_nm = [x_center, y_center, z_top + dipole_offset_nm];
end

if isfield(cfg.dipole, 'orientations')
    dipole_orientations = to_matrix(cfg.dipole.orientations);
else
    dipole_orientations = eye(3);
end
if size(dipole_orientations, 2) ~= 3
    error('dipole.orientations must be N x 3');
end

N_num = to_numeric(cfg.field_sampling.n_acceptors);
x_start_offset_nm = to_numeric(cfg.field_sampling.x_start_offset_nm);
spacing_nm = to_numeric(cfg.field_sampling.spacing_nm);
y_offset_nm = to_numeric(cfg.field_sampling.y_offset_nm);
z_offset_nm = to_numeric(cfg.field_sampling.z_offset_nm);

x_target_min = dipole_position_nm(1) + x_start_offset_nm;
x_target_max = dipole_position_nm(1) + spacing_nm * N_num;
Nx_line = (x_target_max - x_target_min) + 1;

x_line = linspace(x_target_min, x_target_max, Nx_line).';
z_line = (dipole_position_nm(3) + z_offset_nm) * ones(size(x_line));
y_line = dipole_position_nm(2) + y_offset_nm;

lambda_nm = to_numeric(cfg.optical.wavelength_nm);
lambda_m = lambda_nm * 1e-9;
omega = 2 * pi * c0 / lambda_m;

eps_out = to_numeric(cfg.materials.eps_out);
metal_eps_file = char(cfg.materials.metal_eps_file);

epstab = { epsconst(eps_out), epstable(metal_eps_file) };
op = bemoptions('sim', cfg.bem.sim, 'interp', cfg.bem.interp);

p = comparticle(epstab, {pmesh}, [2, 1], 1, op);
pt = compoint(p, dipole_position_nm, op);
emesh = meshfield(p, x_line, y_line, z_line, op, ...
    'mindist', to_numeric(cfg.bem.mindist_nm), ...
    'nmax', to_numeric(cfg.bem.nmax));
bem = bemsolver(p, op);

pref = eps0 * c0^2 / omega^2;

N = numel(x_line);
G = complex(zeros(N, 3, 3));
Ftot = zeros(1, size(dipole_orientations, 1));

for k = 1:size(dipole_orientations, 1)
    dip = dipole(pt, dipole_orientations(k, :), op);
    sig = bem \ dip(p, lambda_nm);

    [tot, ~] = dip.decayrate(sig);
    Ftot(k) = tot;

    e = emesh(sig) + emesh(dip.field(emesh.pt, lambda_nm));
    Eline = squeeze(e);

    G(:, 1, k) = pref * Eline(:, 1);
    G(:, 2, k) = pref * Eline(:, 2);
    G(:, 3, k) = pref * Eline(:, 3);
end

% --- Export data ---
Gxx = G(:, 1, 1);
Gyx = G(:, 2, 1);
Gzx = G(:, 3, 1);
Gxy = G(:, 1, 2);
Gyy = G(:, 2, 2);
Gzy = G(:, 3, 2);
Gxz = G(:, 1, 3);
Gyz = G(:, 2, 3);
Gzz = G(:, 3, 3);

tbl = table(x_line, ...
    real(Gxx), imag(Gxx), real(Gxy), imag(Gxy), real(Gxz), imag(Gxz), ...
    real(Gyx), imag(Gyx), real(Gyy), imag(Gyy), real(Gyz), imag(Gyz), ...
    real(Gzx), imag(Gzx), real(Gzy), imag(Gzy), real(Gzz), imag(Gzz), ...
    'VariableNames', {
    'x_nm', ...
    'Re_Gxx', 'Im_Gxx', 'Re_Gxy', 'Im_Gxy', 'Re_Gxz', 'Im_Gxz', ...
    'Re_Gyx', 'Im_Gyx', 'Re_Gyy', 'Im_Gyy', 'Re_Gyz', 'Im_Gyz', ...
    'Re_Gzx', 'Im_Gzx', 'Re_Gzy', 'Im_Gzy', 'Re_Gzz', 'Im_Gzz' } ...
);

purcell_tbl = table(lambda_nm, Ftot(1), Ftot(2), Ftot(3), ...
    'VariableNames', {'lambda_nm', 'Purcell_x', 'Purcell_y', 'Purcell_z'});

output_file = fullfile(script_dir, char(cfg.output.excel_filename));
writetable(tbl, output_file, 'Sheet', 1);
writetable(purcell_tbl, output_file, 'Sheet', 'Purcell');

function value = to_numeric(value)
    if isstring(value) || ischar(value)
        value = str2double(value);
        return;
    end
    if iscell(value)
        value = cellfun(@to_numeric, value);
        return;
    end
end

function vec = to_row_vector(value, expected_len)
    vec = to_numeric(value);
    vec = reshape(vec, 1, []);
    if numel(vec) ~= expected_len
        error('Expected a vector of length %d.', expected_len);
    end
end

function mat = to_matrix(value)
    if iscell(value)
        mat = cell2mat(cellfun(@(v) reshape(to_numeric(v), 1, []), value, ...
            'UniformOutput', false));
        return;
    end
    mat = to_numeric(value);
    if isvector(mat)
        mat = reshape(mat, 1, []);
    end
end
