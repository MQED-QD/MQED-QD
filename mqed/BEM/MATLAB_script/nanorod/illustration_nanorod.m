% illustration_nanorod: This is program used to illustrate dipole above nanorod.
% Only change the diameter, resolution, and dipole position for the
% interest of user's study. No need to simulate BEM in this program.

% Runtime: Less than a minute in personal laptop.
clear; close all; clc;

% --- materials ---
eps_out = 1.0;                 % vacuum
% eps_in  = epstable('silver.dat');  % metal (or epsconst)

epstab = { epsconst(eps_out), epstable('silver.dat') };

% --- BEM options: retarded simulation in homogeneous medium ---
op = bemoptions('sim','ret','interp','curv');

% --- geometry: choose ONE ---
% 1) Nanorod-like (use a built-in if available in your version)
diameter = 20;  % diameter of nanorod
height = 200;   % height of nanorod

n = [30, 15, 100]; % discertization points. The user can modify this part 
% to observe 'finer' boundary elements on nanorod.

pmesh = trirod( diameter, height, n,'triangles');% (name may differ 
% by MNPBEM version)

% Make the rod horizental
pmesh = rot(pmesh, -90, [0 1 0]); % degrees, axis=x

% Shift nanorod to make the left edge of nanorod at original point, and 
% top of nanorod is at z=0.
x_left = min(pmesh.pos(:,1));
% y_mid = 0.5*(min(pmesh.pos(:,2)) + max(pmesh.pos(:,2)));
z_top = max(pmesh.pos(:,3));
pmesh = shift(pmesh, [-x_left, 0, -z_top]);
% fprintf("nanorod edge:%.2f \n",max(pmesh.pos(:,3)))

% Check if it's a good rod, for a good rod, the z_extent should be close to
% height and x_extent close to diameter.
% fprintf("z extent = %.2f nm\n", max(pmesh.pos(:,3)) - min(pmesh.pos(:,3)));
% fprintf("x extent = %.2f nm\n", max(pmesh.pos(:,1)) - min(pmesh.pos(:,1)));


% --- particle object (outside=1, inside=2) ---
p = comparticle(epstab, {pmesh}, [2,1], 1, op);

% --- dipole ---
z_top = max(pmesh.pos(:,3));
x_left = min(pmesh.pos(:,1));
x_middle = 0.5*(max(pmesh.pos(:,1)+min(pmesh.pos(:,1))));
% put the dipole on the left edge of the nanorod
% r0   = [x_left, 0, z_top + 2];     % nm
r0   = [x_middle, 0, z_top + 2];  %put the dipole 2nm above middle of nanorod.
fprintf("dipole position = %.2f \n", r0)
pdir = [0, 0, 1];

pt1 = compoint(p, r0, op);
dip = dipole(pt1, pdir, op);

% % --- solve ---
% enei = 665;  % nm wavelength in MNPBEM examples
% bem = bemsolver(p, op);
% sig = bem \ dip(p, enei);

% % --- field on a line ---
% x_line = linspace(1,100,100).';
% y_line = 0;
% z_line = 3 * ones(size(x_line));
% 
% emesh = meshfield(p, x_line, y_line, z_line, op, 'mindist', 0.15);
% E = squeeze(emesh(sig) + emesh(dip.field(emesh.pt, enei)));   % N×3

figure('Units','pixels','Position',[100 100 1200 900]); hold on; axis equal; 
% plot particle
plot(pmesh, 'FaceColor',[0.8 0.8 0.85], 'EdgeColor',[0.2 0.2 0.2]); % The 
% face color and EdgeColor here give grey-like color for representing
% silver.

% plot dipole position
plot3(r0(1), r0(2), r0(3), 'ro', 'MarkerFaceColor','r', 'MarkerSize',8);

xlabel('x (nm)'); ylabel('y (nm)'); zlabel('z (nm)');
view(35,20); grid off;
title('Particle mesh + dipole position');

exportgraphics(gcf,sprintf('nanorod_dipole_middle_%.0fnm_2.png',height),'Resolution',800); % used to
% export the .png file of nanorod.

% % draw a little arrow for dipole direction.
% scale = 10; % arrow length in nm for visualization only
% quiver3(r0(1), r0(2), r0(3), pdir(1), pdir(2), pdir(3), scale, ...
%     'Color','r', 'LineWidth',2, 'MaxHeadSize',1.5);


