% test_convergence_nanorod: This is program used to test the electric 
% field convergence of dipole above nanorod of 1000nm length.
% Only change the diameter, number of boundary elements, 
% and dipole position for the interest of user's study.
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
height = 1000;   % height of nanorod

enei = 665; % dipole frequency in wavelength
pdir = [0, 0, 1]; %dipole direction

% BEM resolution
% n_list = {
%     [16,  8, 20];   % very coarse (for your own check only)
%     % [18,  9, 22.5];
%     [20, 10, 25];
%     % [22, 11, 27.5];
%     [24, 12, 30];
%     [28, 14, 35];
%     [32, 16, 40];   % reference
%     [36, 18, 45];   % (optional) super-fine
%     % [40, 20, 50];
% };

% boundary elements list. Change this to get the 'finer' structure for 
% better convergence. Once got the convergence set, used that for later 
% simulation for Green's function
n_list = {
    [20,10,10];
    [20,10,30];
    [20,10,50];
    [20,10,80];
    [20,10,120];
    [20,10,150];
    [20,10,180];
    [20,10,210];
    [20,10,240];
};
test_number = 0;

F_tot_list  = zeros(numel(n_list),1);
Ez_probe    = zeros(numel(n_list),1);
Nfaces = zeros(numel(n_list),1);




for k = 1:numel(n_list)
    n = n_list{k};
    fprintf('Mesh %d: nphi=%d, ntheta=%d, nz=%d\n', k, n);

    % --- build mesh & particle ---
    rodmesh = trirod(diameter, height, n, 'triangles');
    % rotate the rod
    % Make the rod horizental
    rodmesh = rot(rodmesh, -90, [0 1 0]); % degrees, axis=x

    %shift the nanorod to let the left center at x = 0, top at z = 0
    x_left_before = min(rodmesh.pos(:,1));
% y_mid = 0.5*(min(pmesh.pos(:,2)) + max(pmesh.pos(:,2)));
    z_top_before = max(rodmesh.pos(:,3));
 
    rodmesh = shift(rodmesh, [-x_left_before, 0, -z_top_before]);

    % After shift, get the position of x_left and z_top
    x_left = min(rodmesh.pos(:,1));
    x_middle = round(0.5*(min(rodmesh.pos(:,1)+max(rodmesh.pos(:,1)))));
    z_top = max(rodmesh.pos(:,3));
    p = comparticle(epstab, {rodmesh}, [2,1], 1, op);
    Nfaces(k) = size(rodmesh.faces,1);

    % --- dipole + BEM ---
    % r0 = [x_left, 0, z_top + 2];
    r0 = [x_middle, 0, 8];
    pt1 = compoint(p, r0, op);
    probe_pt = [r0(1)+3, 0, r0(3)];  % example: point near dipole
    dip = dipole(pt1, pdir, op);
    bem = bemsolver(p, op);
    sig = bem \ dip(p, enei);

    % Purcell
    [F_tot, F_rad] = dip.decayrate(sig);
    F_tot_list(k) = F_tot;

    % field at probe point
    pt_probe = compoint(p, probe_pt, op);
    emesh = meshfield(p, pt_probe.pos(:,1), pt_probe.pos(:,2), pt_probe.pos(:,3), op);
    e_probe = emesh(sig) + emesh(dip.field(emesh.pt, enei));  % total field
    Ez_probe(k) = e_probe(3);  % z-component

end
% Calculate tolerance
rel_F = abs(diff(F_tot_list)) ./ F_tot_list(1:end-1);
rel_E = abs(diff(Ez_probe)) ./ abs(Ez_probe(1:end-1));

fprintf("rel_F:%.6f \n",rel_F);
fprintf("rel_E:%.6f \n",rel_E);
%% old plot %%
% figure;
% subplot(1,2,1);
% plot(Nfaces, F_tot_list, 'o-');
% xlabel('mesh index'); ylabel('F_{tot}');
% 
% subplot(1,2,2);
% plot(Nfaces, abs(Ez_probe), 'o-');
% xlabel('mesh index'); ylabel('|E_z(probe)|');

% Plot the convergence result:
% figure; clf;
% hold on; box on;
% 
% fig_width = 8;
% fig_height = 6;
% % --- left axis: Purcell factor (Γ / Γ0) ---
% % yyaxis left
% p1 = plot(Nfaces, F_tot_list, 'o-', ...
%     'LineWidth', 1, 'MarkerSize', 4);
% ylabel('$\Gamma / \Gamma_0$', ...
%     'FontSize', 8, 'FontWeight', 'bold','Interpreter','latex');
% % ylim([30,40]);
% 
% % give left axis a color
% set(gca, 'YColor', [0.85 0 0]);      % dark red
% p1.Color = [0.85 0 0];
% 
% % % --- right axis: |E_z| at probe ---
% % yyaxis right
% % p2 = plot(Nfaces, abs(Ez_probe), 's--', ...
% %     'LineWidth', 2, 'MarkerSize', 8);
% % ylabel('|E_z({\bf r}_{probe})|', ...
% %     'FontSize', 18, 'FontWeight', 'bold');
% % 
% % set(gca, 'YColor', [0 0 0.8]);       % dark blue
% % p2.Color = [0 0 0.8];
% 
% % --- common x-axis ---
% xlabel('$N_{elem}$', ...
%     'FontSize', 8, 'FontWeight', 'bold','Interpreter','latex');
% 
% % --- legend, ticks, general style ---
% % legend([p1 p2], ...
% %     {'F_{tot} = \Gamma / \Gamma_0', '|E_z({\bf r}_{probe})|'}, ...
% %     'Location', 'best', 'Box', 'off', 'FontSize', 14);
% legend({'$\Gamma / \Gamma_0$'},'Location','northeast','FontSize',6, ...
%     'Interpreter','latex');
% 
% set(gca, 'FontSize', 8, 'LineWidth', 1);
% set(gcf, 'Units', 'centimeters', 'Position', [5 5 fig_width fig_height]);
% set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [fig_width fig_height]);
% set(gcf, 'PaperPosition', [0 0 fig_width fig_height]);
% 
% % 
% % title('Convergence vs. number of mesh elements', ...
% %     'FontSize', 12, 'FontWeight', 'bold','Interpreter','latex');
% 
% % set(fig,'PaperUnits','centimeters');
% % set(fig,'PaperPosition',[0 0 14 10]);   % [left bottom width height]
% % set(fig,'PaperSize',[14 10]);           % match the paper size to the figure
% print('-dpdf',sprintf(['convergence_purcell_%.0fnm_dip_' ...
%     '%.0fnm_height_%.0fnm_D_%.0fnm_%.0f.pdf'],height,r0(1),r0(3),diameter,test_number));
% 
% Ne  = Nfaces;       % number of elements
% ReE = real(Ez_probe);   % size [nmesh, 1]
% ImE = imag(Ez_probe);
% 
% figure; hold on;
% set(gcf, 'Units','centimeters', 'Position',[5 5 14 10]);  % nice compact size
% 
% % --- Left axis: Re(Ez) ---
% % yyaxis left;
% plot(Ne, ReE, 'o-','LineWidth',1.5, 'MarkerSize',6);
% xlabel('$N_{elem}$','FontSize',8,'FontWeight','bold', ...
%     'Interpreter','latex');
% ylabel('Re${E_z}$ ','FontSize',8,'FontWeight','bold', ...
%     'Interpreter','latex');
% set(gca,'YColor',[0 0 0.6]);   % dark blue-ish for left axis
% legend({'Re($E_z$)'},'Location','northeast','FontSize',6, ...
%     'Interpreter','latex');
% set(gcf, 'Units', 'centimeters', 'Position', [5 5 fig_width fig_height]);
% set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [fig_width fig_height]);
% set(gcf, 'PaperPosition', [0 0 fig_width fig_height]);
% 
% % optional: zero line for the real part
% yline(0,'k:');
% print('-dpdf',sprintf(['convergence_real_field_%.0fnm_dip_%.0fnm_' ...
%     'height_%.0fnmD_%.0fnm_%.0f.pdf'],height ...
%     ,r0(1),r0(3),diameter,test_number));
% figure; hold on;
% % set(gcf, 'Units','centimeters', 'Position',[5 5 14 10]);  % nice compact size
% % --- Right axis: Im(Ez) ---
% % yyaxis right;
% plot(Ne, ImE, 's--','LineWidth',1, 'MarkerSize',4);
% ylabel('Im${E_z}$ ','FontSize',8,'FontWeight','bold', ...
%     'Interpreter','latex');
% set(gca,'YColor',[0.6 0 0]);   % dark red-ish for right axis
% % --- set Y range for better visualization ---%
% current_mean = ImE(3);
% ylim([current_mean*0.8,current_mean*1.2]);
% 
% % --- Cosmetics ---
% set(gca,'FontSize',6,'Box','on');
% set(gcf, 'Units', 'centimeters', 'Position', [5 5 fig_width fig_height]);
% set(gcf, 'PaperUnits', 'centimeters', 'PaperSize', [fig_width fig_height]);
% set(gcf, 'PaperPosition', [0 0 fig_width fig_height]);
% legend({'Im($E_z$)'},'Location','northeast','FontSize',6, ...
%     'Interpreter','latex');
% 
% % title('Convergence of E_z(r_{probe}) with mesh refinement', ...
% %       'FontSize',14,'FontWeight','bold','Interpreter','latex');
% print('-dpdf',sprintf(['convergence_imag_field_%.0fnm_dip_%.0fnm_' ...
%     'height_%.0fnmD_%.0fnm_%.0f.pdf'],height ...
%     ,r0(1),r0(3),diameter,test_number));
%% ------------ Save convergence data -----------------
Ne   = Nfaces(:);          % number of elements
ReE  = real(Ez_probe(:));  % real(Ez) at probe
ImE  = imag(Ez_probe(:));  % imag(Ez) at probe
Purcell = F_tot_list(:);   % Γ/Γ0

convTable = table(Ne, Purcell, ReE, ImE, ...
    'VariableNames', {'Nelem', 'Purcell', 'ReEz', 'ImEz'});

% make a small data directory
dataDir = 'convergence_data';
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

baseName = sprintf('conv_h%gnm_pos%gnm_D%gnm_dip%.0fnm', r0(3), ...
    r0(1),diameter,enei);
csvFile  = fullfile(dataDir, [baseName '.csv']);

writetable(convTable, csvFile);
fprintf('Convergence data saved to CSV: %s\n', csvFile);


%% ------------ Plotting (ACS-style) -----------------
Ne   = Nfaces;          % number of elements
ReE  = real(Ez_probe);  % [nmesh, 1]
ImE  = imag(Ez_probe);

% === style parameters (tweak here) ===
figW_cm      = 8.5;   % single-column width ~3.35 in
figH_cm      = 6.0;   % height
fs_label     = 8;     % axis-label font size
fs_tick      = 7;     % tick font size
fs_legend    = 7;     % legend font size
lw_axis      = 0.75;  % axis line width
lw_plot      = 0.9;   % curve line width
ms_plot      = 4;     % marker size

set(0,'DefaultAxesFontName','Helvetica');  % typical journal font

make_acs_figure = @(h) set(h, ...
    'Units','centimeters', ...
    'Position',[2 2 figW_cm figH_cm], ...
    'PaperUnits','centimeters', ...
    'PaperSize',[figW_cm figH_cm], ...
    'PaperPosition',[0 0 figW_cm figH_cm]);

%% 1) Purcell factor Γ/Γ0
fig1 = figure; hold on; box on;
plot(Ne, F_tot_list, 'o-', ...
    'LineWidth', lw_plot, ...
    'MarkerSize', ms_plot);

xlabel('$N_{\mathrm{elem}}$', ...
    'Interpreter','latex', ...
    'FontSize', fs_label);
ylabel('$\Gamma / \Gamma_0$', ...
    'Interpreter','latex', ...
    'FontSize', fs_label);

set(gca, ...
    'FontSize', fs_tick, ...
    'LineWidth', lw_axis);

legend({'$\Gamma/\Gamma_0$'}, ...
    'Interpreter','latex', ...
    'FontSize', fs_legend, ...
    'Location','best', ...
    'Box','off');
Purcell_mean = F_tot_list(round(numel(F_tot_list)/2));
ylim = ([0.7*Purcell_mean, 1.3*Purcell_mean]);

make_acs_figure(fig1);
print(fig1, sprintf('conv_purcell_h%gnm_pos%gnm_D%gnm_dip%gnm.pdf', r0(3), ...
    r0(1),diameter, enei), ...
    '-dpdf','-r600');

%% 2) Real part of Ez at probe
fig2 = figure; hold on; box on;
plot(Ne, ReE, 'o-', ...
    'LineWidth', lw_plot, ...
    'MarkerSize', ms_plot);

xlabel('$N_{\mathrm{elem}}$', ...
    'Interpreter','latex', ...
    'FontSize', fs_label);
ylabel('$(E_z)$', ...
    'Interpreter','latex', ...
    'FontSize', fs_label);

yline(0,'k:');  % zero reference

set(gca, ...
    'FontSize', fs_tick, ...
    'LineWidth', lw_axis);

legend({'Re$(E_z)$'}, ...
    'Interpreter','latex', ...
    'FontSize', fs_legend, ...
    'Location','best', ...
    'Box','off');

make_acs_figure(fig2);
print(fig2, sprintf('conv_ReEz_h%gnm_pos_%gnm_D%gnm_dip%gnm.pdf', r0(3), ...
    r0(1),diameter,enei), ...
    '-dpdf','-r600');

%% 3) Imag part of Ez at probe
fig3 = figure; hold on; box on;
plot(Ne, ImE, 's--', ...
    'LineWidth', lw_plot, ...
    'MarkerSize', ms_plot);

xlabel('$N_{\mathrm{elem}}$', ...
    'Interpreter','latex', ...
    'FontSize', fs_label);
ylabel('$(E_z)$', ...
    'Interpreter','latex', ...
    'FontSize', fs_label);

% optional: zoom around typical value
meanIm = ImE(3);
ylim([0.8*meanIm, 1.2*meanIm]);

set(gca, ...
    'FontSize', fs_tick, ...
    'LineWidth', lw_axis);

legend({'Im$(E_z)$'}, ...
    'Interpreter','latex', ...
    'FontSize', fs_legend, ...
    'Location','best', ...
    'Box','off');

make_acs_figure(fig3);
print(fig3, sprintf('conv_ImEz_h%gnm_D%gnm_dip%gnm.pdf', height, ...
    diameter,enei), ...
    '-dpdf','-r600');

