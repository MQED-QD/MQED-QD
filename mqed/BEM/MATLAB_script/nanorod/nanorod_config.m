function cfg = nanorod_config()
cfg.nanorod.diameter_nm = 20;
cfg.nanorod.length_nm = 1000;
cfg.nanorod.mesh_n = [20,10,150];
cfg.nanorod.rotate_deg = -90;
cfg.nanorod.rotate_axis = [0, 1, 0];
cfg.nanorod.align_left_edge_x0 = true;
cfg.nanorod.align_top_z0 = true;

cfg.dipole.position_mode = 'above_center';
cfg.dipole.offset_nm = 2;
cfg.dipole.position_nm.x = 0;
cfg.dipole.position_nm.y = 0;
cfg.dipole.position_nm.z = 2;
cfg.dipole.orientations = [
    1, 0, 0;
    0, 1, 0;
    0, 0, 1
];

cfg.field_sampling.n_acceptors = 30;
cfg.field_sampling.x_start_offset_nm = 1;
cfg.field_sampling.spacing_nm = 8;
cfg.field_sampling.y_offset_nm = 0;
cfg.field_sampling.z_offset_nm = 0;

cfg.optical.wavelength_nm = 665;

cfg.materials.eps_out = 1.0;
cfg.materials.metal_eps_file = 'silver.dat';

cfg.bem.sim = 'ret';
cfg.bem.interp = 'curv';
cfg.bem.mindist_nm = 0.15;
cfg.bem.nmax = 3000;

cfg.output.excel_filename = 'dipole_silver_nanorod_GF.xlsx';
end
