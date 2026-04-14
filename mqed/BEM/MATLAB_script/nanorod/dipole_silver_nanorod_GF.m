%dipole_silver_nanorod_GF.
% This is the script to generate the dyadic Green's Function 
% tensor elements. 
% Steps:
% 1. Give material informationand build nanorod.
% 2. Define the SI unit constant and construct pref*E, which is the value
% of Green's function.
% 3. Define the range of user's interest. Here we are interested in the
% acceptor at the right side of dipole, so we set the acceptor x_target_min
% and x_target_max for the points we want to extract.
% 4. Simulate BEM and extract the Purcell factor and electric field.
% 5. Store the data in xlsx file.
clear; close all; clc;

%% --- materials info ---
eps_out = 1.0;                 % vacuum
% eps_in  = epstable('silver.dat');  % metal (or epsconst)
% eps_in = -80;

epstab = { epsconst(eps_out), epstable('silver.dat') }; %vacuum outside,
% epstab = { epsconst(eps_out), epsconst(eps_in) }; %vacuum outside
% silver inside.

%% --- BEM options: retarded simulation in homogeneous medium ---
op = bemoptions('sim','ret','interp','curv');

%% --- geometry: choose ONE ---
% 1) Nanorod-like (use a built-in if available in your version)
diameter = 20;  % diameter of nanorod
height = 1000;   % height of nanorod
% num_points = 100;  % number of discertization points
n = [20, 10, 150]; % discertization points, converged for this simulation.
% If change the nanorod length or dipole height above nanorod, do
% convergence test before choosing appropriate parameters.

pmesh = trirod( diameter, height, n, 'triangles' ); % (name may differ by MNPBEM version)

% Make the rod horizental
pmesh = rot(pmesh, -90, [0 1 0]); % degrees, axis=x

% Shift nanorod.
x_left_before = min(pmesh.pos(:,1));
% y_mid = 0.5*(min(pmesh.pos(:,2)) + max(pmesh.pos(:,2)));
z_top_before = max(pmesh.pos(:,3));
pmesh = shift(pmesh, [-x_left_before, 0, -z_top_before]);
% fprintf("nanorod edge:%.2f \n",max(pmesh.pos(:,3)))

%% --- particle object (outside=1, inside=2) ---
p = comparticle(epstab, {pmesh}, [2,1], 1, op);

%% --- dipole ---
z_top = max(pmesh.pos(:,3));
z_top = round(z_top);
x_middle = 0.5*(min(pmesh.pos(:,1)+max(pmesh.pos(:,1))));
x_middle = round(x_middle); % make the middle point integer 
% put the dipole on the left edge of the nanorod
height = 8; % the height of dipole above nanorod in nm.
r0   = [x_middle, 0, z_top + height];     %  dipole position in nm.
pt1 = compoint(p, r0, op);

%% Electric field calculation range, unit:nm
x_target_min = r0(1)+1; % The first response point eletric field.
N_num=30;  % we are studying N_num acceptor.
x_target_max = r0(1)+8*N_num; % The last point is N_num-th acceptor position.
z_target_val = r0(3);  % same as dipole height above surface.
Nx_line = (x_target_max-x_target_min)+1;             % number of x samples

%  wavelength corresponding to transition dipole energy
enei = 665; %in nm

% Output name, format:'dipole_material_nanorod_dipole position(z-axis
% here)_GF_wavelength_middle.xlsx', GF tells this program output Green's 
% function elements, middle tells we are putting dipole in the middle of
% nanorod.
test_number = 1;  %use for the name of file.
xlsx_name = sprintf(['dipole_silver_nanorod_GF_%.0fnm_pos_%.0f' ...
    '_height_%.0fnm_D_%.0fnm_%.0f.xlsx'], ...
                    enei,r0(1),r0(3),diameter,test_number);
% xlsx_name = sprintf(['dipole_const_nanorod_GF_%.0fnm_pos_%.0f' ...
%     '_height_%.0f_%.0f.xlsx'], ...
%                     enei,r0(1),r0(3),test_number);

% ---- constants (SI) ----
eps0 = 8.8541878128e-12;   % F/m
c0   = 299792458;          % m/s
lambda_m = enei * 1e-9;  %nm
omega = 2*pi*c0 / lambda_m;  % s^-1
pref = eps0 * c0^2 / omega^2; % prefactor used to calculate GF elements.

% ----- line points you want -----
x_line = linspace(x_target_min, x_target_max, Nx_line).';
z_line = z_target_val * ones(size(x_line));
y_line = 0;

N = numel(x_line);

% allocate dyadic
G = complex(zeros(N, 3, 3));
Ftot = zeros(1,3);

% dipole basis
pdirs = eye(3);

% solver once
bem = bemsolver(p, op);

% meshfield once
emesh = meshfield(p, x_line, y_line, z_line, op, 'mindist', 0.15, 'nmax', 3000);

for j = 1:3
    dip = dipole(pt1, pdirs(j,:), op);
    sig = bem \ dip(p, enei);

    [tot, ~] = dip.decayrate(sig);
    Ftot(j) = tot;

    % TOTAL field at line points
    e = emesh(sig) + emesh(dip.field(emesh.pt, enei));   % <-- include sig here for non-planar situation.
    Eline = squeeze(e);   % should become N×3

    G(:,1,j) = pref * Eline(:,1);
    G(:,2,j) = pref * Eline(:,2);
    G(:,3,j) = pref * Eline(:,3);
end

F_xx = Ftot(1); F_yy = Ftot(2); F_zz = Ftot(3);


% helper to split real/imag
Re = @(v) real(v);
Im = @(v) imag(v);

T = table(x_line, ...
    Re(G(:,1,1)), Im(G(:,1,1)), Re(G(:,1,2)), Im(G(:,1,2)), Re(G(:,1,3)), Im(G(:,1,3)), ...
    Re(G(:,2,1)), Im(G(:,2,1)), Re(G(:,2,2)), Im(G(:,2,2)), Re(G(:,2,3)), Im(G(:,2,3)), ...
    Re(G(:,3,1)), Im(G(:,3,1)), Re(G(:,3,2)), Im(G(:,3,2)), Re(G(:,3,3)), Im(G(:,3,3)), ...
    'VariableNames', { ...
      'x_nm', ...
      'Re_Gxx','Im_Gxx','Re_Gxy','Im_Gxy','Re_Gxz','Im_Gxz', ...
      'Re_Gyx','Im_Gyx','Re_Gyy','Im_Gyy','Re_Gyz','Im_Gyz', ...
      'Re_Gzx','Im_Gzx','Re_Gzy','Im_Gzy','Re_Gzz','Im_Gzz'});

Tself = table(enei, F_xx, F_yy, F_zz, ...
  'VariableNames', {'lambda_nm','Purcell_x','Purcell_y','Purcell_z'});

writetable(T, xlsx_name, 'Sheet', 'DyadicG');
writetable(Tself, xlsx_name, 'Sheet', 'G_self');