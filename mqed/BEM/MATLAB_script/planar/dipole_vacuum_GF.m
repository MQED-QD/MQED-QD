% dipole_vacuum_GF: This program is used to collect electric field in the
% vacuum. The result will be used to compare with analytical solution in
% Python program from MQED-QD and calibrate the dipole moment intensity in
% BEM with unknown unit. 
% User do not need to change any parameters except the wavelength of dipole
% and dipole position for different system of studying.

clear; close all; clc;
%%  initialization
% Upper medium (z>0)
n_top  = 1.0;              % e.g. 1.0 for air
eps_top = n_top^2;

% Substrate (z<0)
n_sub  = 1.0;              % example; change to your substrate index
eps_sub = n_sub^2;

%  location of interface of substrate
ztab = 0;

% Dipole settings
r0   = [0, 0, 2];          % dipole position (nm)
pdir = [0, 0, 1];          % orientation (unit vector)

% Electric field calculation range, unit:nm
% we choose [x_target_min, x_target_max] multiple point values and use 
% lease square to get more accurate result of dipole moment intensity.
x_target_min = 7;
x_target_max = 100;
z_target_val = r0(3)+1; % we shift the height of z-axis of acceptor by 1nm 
% to avoid zero values due to symmetry.
Nx_line = x_target_max- x_target_min +1;             % number of x samples


%  wavelength corresponding to transition dipole energy
enei = 300;  %in nm

% Output format: 'dipole_vacuum_height of dipole_GF_x_target_min_wavelength
% nm.xlsx' GF means used to calibrate Green's Function in later process.
xlsx_name = sprintf('dipole_vacuum_%.0fnm_GF_%.0fnm_%.0fnm.xlsx', ...
                    r0(3), x_target_min, enei);

%% Layer + option
%  table of dielectric functions
epstab = { epsconst( eps_top ), epsconst( eps_top ), epsconst( eps_sub ) }; 
%  default options for layer structure
opt = layerstructure.options; 
%  set up layer structure
layer = layerstructure( epstab, [ 1, 3 ], ztab, opt );
%  options for BEM simulation
op = bemoptions( 'sim', 'ret', 'interp', 'curv' , 'layer', layer );

%  nanosphere with finer discretization at the bottom
p = flip( trisphere( 32, 1 ), 1 );
%  place nanosphere 1 nm above substrate
p = shift( p, [ 0, 0, (- min( p.pos( :, 3 ) ) + 6 + ztab) ] ); 

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

%%  BEM simulation
%  dipole excitation
dip = dipole( pt1, pdir, op );
%  set up BEM solver
bem = bemsolver( p, op );
% %  surface charge
% sig = bem \ dip( p, enei );
% 
% %% Purcell factor
% [F_tot, F_rad] = dip.decayrate(sig);
% 
% F_nrad = F_tot - F_rad;   % in lossless substrate this should be ~0

%%  computation of electric field
% %  mesh for calculation of electric field
% [ x, z ] = meshgrid( linspace( - 8, 8, 81 ) );
%  object for electric field
%    MINDIST controls the minimal distance of the field points to the
%    particle boundary

% ----- line points you want -----
x_line = linspace(x_target_min, x_target_max, Nx_line).';
z_line = z_target_val * ones(size(x_line));   % fixed z
y_line = 0;

emesh = meshfield( p, x_line, y_line, z_line, op, 'mindist', 0.15, 'nmax', 3000 );
%  induced and incoming electric field
e = emesh( dip.field( emesh.pt, enei ) );
%  norm of electric field
ee = sqrt( dot( e, e, 3 ) );



% make it N×3
Eline = squeeze(e);        % <--- THIS is where squeeze actually helps

Ex = Eline(:,1);
Ey = Eline(:,2);
Ez = Eline(:,3);
Enorm = sqrt( Ex.*conj(Ex) + Ey.*conj(Ey) + Ez.*conj(Ez) );

%% ---------------- Pack into tables + write XLSX -------------------------
% Tpurcell = table( ...
%     enei, F_tot, ...
%     'VariableNames', {'lambda_nm','F_tot'} );

Tline = table( ...
    x_line, ...
    real(Ex), imag(Ex), ...
    real(Ey), imag(Ey), ...
    real(Ez), imag(Ez), ...
    'VariableNames', { ...
        'x_nm', ...
        'Re_Ex','Im_Ex', ...
        'Re_Ey','Im_Ey', ...
        'Re_Ez','Im_Ez', ...
        } );

% writetable(Tpurcell, xlsx_name, 'Sheet', 'Purcell');
writetable(Tline,    xlsx_name, 'Sheet', 'FieldLine');

fprintf('Wrote: %s\n', xlsx_name);
% fprintf('Purcell: F_tot=%.6g, F_rad=%.6g, F_nrad=%.6g\n', F_tot, F_rad, F_nrad);


