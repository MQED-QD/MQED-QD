%  dipole_silver_planar_GF - Electric field above infinite silver layer
% This is the script to generate the dyadic Green's Function 
% tensor elements. We use this to verify the BEM accuracy by comparing 
% with Fresnel/Sommerfeld integral implemented in Python.
% 
% Steps:
% 1.Build layer, z>0: air; z<0: silver.
% 2.Give the location of substrate and dipole position, direction. 
% 3.Build imaginary comparticle object used to calculate Purcell factor.
% It should be far away from our substrate. If some wavelength gives error,
% try to put the comparticle even further away from our substrate and
% dipole.
% 4.BEM simulation of dipole in three directions and extract Purcell factor
% as well as electric field to calculate Dyadic Green's function elements.
% 5. Store the data and export for further process.

clear; close all; clc;
%%  initialization
% Upper medium (z>0)
n_top  = 1.0;              % e.g. 1.0 for air
eps_top = n_top^2; % permitivity for the air.

%  location of interface of substrate
ztab = 0; %Don't change here, unless necessary, read MNPBEM document.

% Dipole settings
r0   = [0, 0, 2];          % dipole position (nm)

% Electric field calculation range, unit:nm
x_target_min = 1;  % the first point of acceptor
x_target_max = 300; % the last point of acceptor
z_target_val = r0(3); % the height of the acceptor, keep it same as dipole here.
Nx_line = x_target_max;             % number of x samples

% Wavelength corresponding to transition dipole energy
enei = 665; % unit:nm

% Output name, format: 'dipole_material_dipole position(z-axis
% here)_GF_wavelength.xlsx', GF tells this program output Green's function
% elements.
xlsx_name = sprintf('dipole_silver_planar_height_%.0fnm_GF_%.0fnm.xlsx', ...
                    r0(3),  enei);

%% Layer + option
%  table of dielectric functions
epstab = { epsconst( eps_top ), epsconst( eps_top ), epstable( 'silver.dat') }; 
%  default options for layer structure
opt = layerstructure.options; 
%  set up layer structure
layer = layerstructure( epstab, [ 1, 3 ], ztab, opt );
%  options for BEM simulation
op = bemoptions( 'sim', 'ret', 'interp', 'curv' , 'layer', layer );

%  nanosphere with finer discretization at the bottom
p = flip( trisphere( 32, 1 ), 1 );
%  place nanosphere 20 nm above substrate so that it does not affect our
%  simulation of planar surface.
p = shift( p, [ 0, 0, (- min( p.pos( :, 3 ) ) + 20 + ztab) ] ); 

%  set up COMPARTICLE objec
p = comparticle( epstab, { p }, [ 2, 1 ], 1, op );

%%  tabulated Green function
%  For the retarded simulation we first have to set up a table for the
%  calculation of the reflected Green function.  This part is usually slow
%  and we thus compute GREENTAB only if it has not been computed before.
%
%  For setting up the table for the reflected Green function, we need to
%  provide all points for which we will compute it.  As we will compute the
%  nearfield enhancement above and below the substrate interface using the
%  MESHFIELD class, we here set up a COMPOINT object.  Note that the
%  MESHFIELD object must be initialized later because it needs the
%  precomputed Green function table.
%
%  As we also want to compute the electromagnetic fields of the dipole, we
%  have to register in TABSPACE the dipole positions as both source and
%  observation points.

%  dipole position
pt1 = compoint( p, r0, op );
%  mesh for calculation of electric field
[ x, z ] = meshgrid( linspace( x_target_min, x_target_max, Nx_line ) );
%  make compoint object
%    it is important that COMPOINT receives the OP structure because it has
%    to group the points within the layer structure
pt2 = compoint( p, [ x( : ), 0 * x( : ), z( : ) ], op );


%  tabulated Green functions
%    For the retarded simulation we first have to set up a table for the
%    calculation of the reflected Green function.  This part is usually
%    slow and we thus compute GREENTAB only if it has not been computed
%    before.
if ~exist( 'greentab', 'var' ) || ~greentab.ismember( layer, enei, { p, pt1 }, pt2 )
  %  automatic grid for tabulation
  %    we use a rather small number NZ for tabulation to speed up the
  %    simulations
  tab = tabspace( layer, { p, pt1 }, pt2, 'nz', 40 );
  %  Green function table
  greentab = compgreentablayer( layer, tab );
  %  precompute Green function table
  greentab = set( greentab, enei, op, 'waitbar', 0 );
end
op.greentab = greentab;

%% Get Dyadic Green's Function

% ---- constants (SI) ----
eps0 = 8.8541878128e-12;   % F/m
c0   = 299792458;          % m/s
lambda_m = enei * 1e-9;  % nm
omega = 2*pi*c0 / lambda_m;   % s^-1
pref = eps0 * c0^2 / omega^2; %prefactor of calculating GF from electric field.

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
    e = emesh(dip.field(emesh.pt, enei));   % <-- include sig here for non-planar situation.
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




