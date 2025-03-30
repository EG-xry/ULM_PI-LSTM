%% PALA_ImageGeneration_fromCSV.m
% Script for image generation from final tracking CSV file.
%
% The CSV file (tracks.csv) must be organized as columns:
%   track_id, point_index, x, y, t
% with the first row containing the header names.
%
% This script reads the CSV, groups the data by track, converts each track 
% into the expected format, and then generates the following images:
%   - ULM intensity display (MatOut)
%   - ULM intensity display with axial flow direction (MatOut_zdir)
%   - Velocity magnitude rendering (MatOut_vel)
%
% Required functions: ULM_Track2MatOut, WriteTif, imgaussfilt, brighten.
%
% Created by [Your Name] on [Date]

clc;
clear;
close all;

%% Set up output folder and prefix definitions
workingdir = fullfile(pwd, 'ExampleImages');
if ~exist(workingdir, 'dir')
    mkdir(workingdir);
end
filename = 'PALA_InVivoRatBrain_';  % Prefix for exported image files

%% ULM Parameters (adapt these if needed)
% In the original processing, the pixel scale is set to 1 and the frame rate to 1000 Hz.
% Here, default parameters are defined and the ultrasound wavelength is computed from a 15MHz transmit.
res = 10;                   % Resolution factor of the localization (1/res = final pixel size in super-res grid)
ScaleOfPixel = [1, 1];      % [vertical, horizontal] pixel size (in lambda units)
framerate = 1000;           % Imaging frame rate in Hz
TWFreq = 15;                % Transmit frequency in MHz
lambda = 1540/(TWFreq*1e3); % Ultrasound wavelength in mm

%% Load tracking data from CSV file
% CSV must contain columns: track_id, point_index, x, y, t
tracksTable = readtable('tracks.csv');
requiredVars = {'track_id','point_index','x','y','t'};
if ~all(ismember(requiredVars, tracksTable.Properties.VariableNames))
    error('The CSV file does not contain all the required columns.');
end

% Group data by track_id. For each track, sort by point_index
% and create a matrix with columns: [x, y, t, velocity].
% (The velocity column is set to zero as a placeholder.)
trackIDs = unique(tracksTable.track_id);
Track_tot = cell(length(trackIDs),1);
for i = 1:length(trackIDs)
    idx = tracksTable.track_id == trackIDs(i);
    subTable = sortrows(tracksTable(idx,:), 'point_index');
    % Construct a matrix: columns [x, y, t, dummy_velocity]
    Track_tot{i} = [subTable.x, subTable.y, subTable.t, zeros(height(subTable), 1)];
end

%% Determine ULM size from the tracking data (used to create the output grid)
allTracks = cell2mat(Track_tot);
max_x = max(allTracks(:,1));
max_y = max(allTracks(:,2));
% ULM.size: [vertical (rows), horizontal (cols), number of frames (dummy here)]
ULM.size = [ceil(max_y)+20, ceil(max_x)+20, 100];

%% Build the ULM structure
ULM.numberOfParticles = 90;
ULM.scale = [ScaleOfPixel, 1/framerate];   % [vertical, horizontal, dt]
ULM.res = res;
ULM.SVD_cutoff = [5, ULM.size(3)];  
ULM.max_linking_distance = 2;
ULM.min_length = 15;
ULM.fwhm = [3, 3];
ULM.max_gap_closing = 0;
ULM.interp_factor = 1/res;
ULM.LocMethod = 'Radial';
ULM.ButterCuttofFreq = [50, 249];
ULM.parameters.NLocalMax = 3;
ULM.lambda = lambda;
ULM.velocity_tol = 1.0;

% Compute the super-resolved pixel size and SR grid dimensions
ULM.SRscale = ULM.scale(1) / ULM.res;  
ULM.SRsize  = round(ULM.size(1:2) .* ULM.scale(1:2) / ULM.SRscale);

%% Process tracking data for image generation
% Convert track coordinates into the "super-resolved" pixel coordinates.
% The original routine adds an offset [1 1 0 0] and divides spatial
% coordinates by ULM.SRscale.
Track_matout = cellfun(@(x) (x + repmat([1 1 0 0], size(x,1), 1)) ./ ...
    repmat([ULM.SRscale, ULM.SRscale, 1, 1], size(x,1), 1), Track_tot, 'UniformOutput', false);

% Define spatial grid for the output images
llz = (0:ULM.SRsize(1)) * ULM.SRscale; % vertical coordinate grid
llx = (0:ULM.SRsize(2)) * ULM.SRscale; % horizontal coordinate grid

%% Create MatOut and related images using ULM_Track2MatOut
% These functions are assumed to be available.
MatOut    = ULM_Track2MatOut(Track_matout, ULM.SRsize + [1 1]);
MatOut_zdir = ULM_Track2MatOut(Track_matout, ULM.SRsize + [1 1], 'mode', '2D_vel_z');
MatOut_vel  = ULM_Track2MatOut(Track_matout, ULM.SRsize + [1 1], 'mode', '2D_velnorm');
MatOut_vel  = MatOut_vel * ULM.lambda;  % Convert velocity into [mm/s]

%% Visualization - Figure 1: ULM intensity display
close all;
fig = figure('Visible', 'on');
clf(fig);
set(fig, 'Position', [652 393 941 585]);

IntPower = 1/3;
SigmaGauss = 0;
imHandle = imagesc(llx, llz, MatOut.^IntPower);
axis image;
if SigmaGauss > 0
    imHandle.CData = imgaussfilt(imHandle.CData, SigmaGauss);
end
title('ULM intensity display');
colormap(gca, gray(128));
clbar = colorbar;
caxis(caxis * 0.8);
clbar.Label.String = 'number of counts';
% Adjust tick labels to counteract the intensity power
clbar.TickLabels = round(clbar.Ticks.^(1/IntPower), 1);
xlabel('\lambda');
ylabel('\lambda');
ca = gca;
ca.Position = [.05 .05 .8 .9];
BarWidth = round(1 / (ULM.SRscale * lambda));
% Embed a saturation bar in the image
imHandle.CData(end-50+(0:3), 60+(0:BarWidth)) = max(caxis);
drawnow;
pause(0.1);
if ~isgraphics(fig)
    warning('Figure handle lost. Recreating figure.');
    fig = figure('Visible', 'on');
end
exportFile = fullfile(workingdir, [filename, 'example_MatOut.png']);
print(fig, exportFile, '-dpng', '-r300');

%% Visualization - Figure 2: ULM intensity display with axial flow direction
figure(2); clf;
set(gcf, 'Position', [652 393 941 585]);
velColormap = cat(1, flip(flip(hot(128),1),2), hot(128)); % custom colormap
velColormap = velColormap(5:end-5, :); % trim white parts
IntPower = 1/4;
imHandle = imagesc(llx, llz, (MatOut).^IntPower .* sign(imgaussfilt(MatOut_zdir, 0.8)));
imHandle.CData = imHandle.CData - sign(imHandle.CData)/2;
axis image;
title('ULM intensity display with axial flow direction');
colormap(gca, velColormap);
caxis([-1 1] * max(caxis) * 0.7);
clbar = colorbar;
clbar.Label.String = 'Count intensity';
ca = gca;
ca.Position = [.05 .05 .8 .9];
BarWidth = round(1 / (ULM.SRscale * lambda));
imHandle.CData(end-50+(0:3), 60+(0:BarWidth)) = max(caxis);

% Store current colormap and caxis limits BEFORE printing/exporting
currentColormap = get(gca, 'Colormap');
currentCaxis = caxis;

print(gcf, fullfile(workingdir, [filename, 'example_MatOut_zdir.png']), '-dpng', '-r750');

% Use stored values in WriteTif instead of passing graphics handles
WriteTif(imHandle.CData, currentColormap, ...
    fullfile(workingdir, [filename, 'example_MatOut_zdir.tif']), ...
    'caxis', currentCaxis, 'Overwrite', 1);

%% Visualization - Figure 3: Velocity magnitude rendering
% Determine the maximum displayed velocity (in mm/s)
vmax_disp = ceil(quantile(MatOut_vel(MatOut_vel > 0), 0.98) / 10) * 10;
figure(3); clf;
set(gcf, 'Position', [652 393 941 585]);
clbsize = [180, 50];
Mvel_rgb = MatOut_vel / vmax_disp;  % Normalize velocity values
% Add a colorbar to the upper-left corner
Mvel_rgb(1:clbsize(1), 1:clbsize(2)) = repmat(linspace(1, 0, clbsize(1))', 1, clbsize(2));
Mvel_rgb = Mvel_rgb.^(1/1.5);
Mvel_rgb(Mvel_rgb > 1) = 1;
Mvel_rgb = imgaussfilt(Mvel_rgb, 0.5);
Mvel_rgb = ind2rgb(round(Mvel_rgb*256), jet(256));  % Convert normalized values into an RGB image

% Create a shadow image from MatOut to modulate the velocity display
MatShadow = MatOut;
MatShadow = MatShadow ./ max(MatShadow(:) * 0.3);
MatShadow(MatShadow > 1) = 1;
MatShadow(1:clbsize(1), 1:clbsize(2)) = repmat(linspace(0, 1, clbsize(2)), clbsize(1), 1);
Mvel_rgb = Mvel_rgb .* (MatShadow.^IntPower);
Mvel_rgb = brighten(Mvel_rgb, 0.4);
BarWidth = round(1 / (ULM.SRscale * lambda));
Mvel_rgb(end-50+(0:3), 60+(0:BarWidth), :) = 1;
imshow(Mvel_rgb, 'XData', llx, 'YData', llz);
axis on;
title(['Velocity magnitude (0-' num2str(vmax_disp) 'mm/s)']);
ca = gca;
ca.Position = [.05 .05 .8 .9];
print(gcf, fullfile(workingdir, [filename, 'example_VelMorm.png']), '-dpng', '-r750');
imwrite(Mvel_rgb, fullfile(workingdir, [filename, 'example_VelMorm.tif']));

fprintf('Image generation from CSV completed.\n');

%% Plain Brain Map Generation from CSV Tracking Data
% This script reads the CSV file (tracks.csv), which must contain the columns:
%   track_id, point_index, x, y, t
% It then groups the data by track_id and generates a plain brain map.
%    - Background: white
%    - Tracks: connected with red lines (extremely thin)
%    - Points: red circles (smaller)
%    - Graph: x and y coordinates labeled accordingly
%
% Created by [Your Name] on [Date]

clc;
clear;
close all;

%% Set up output folder and file prefix definitions
workingDir = fullfile(pwd, 'BrainMapImages');
if ~exist(workingDir, 'dir')
    mkdir(workingDir);
end
filename = 'PlainBrainMap_';

%% Load tracking data from CSV file
% CSV must contain columns: track_id, point_index, x, y, t
tracksTable = readtable('tracks.csv');
requiredVars = {'track_id', 'point_index', 'x', 'y', 't'};
if ~all(ismember(requiredVars, tracksTable.Properties.VariableNames))
    error('The CSV file does not contain all the required columns.');
end

%% Group data by track_id
trackIDs = unique(tracksTable.track_id);

%% Plot the brain map
figure('Color','w');  % White background
hold on;
grid on;
axis equal;
xlabel('X Coordinate');
ylabel('Y Coordinate');
title('Plain Brain Map of Tracks');

% Loop through each track and plot the connecting line and points
for i = 1:length(trackIDs)
    % Extract and sort this track's data by point_index
    idx = tracksTable.track_id == trackIDs(i);
    subTable = sortrows(tracksTable(idx, :), 'point_index');
    
    % Get x and y coordinates
    xPoints = subTable.x;
    yPoints = subTable.y;
    
    % Plot the track: red line connecting the points (extremely thin)
    plot(xPoints, yPoints, 'r-', 'LineWidth', 0.5);
    
    % Plot individual points as red circles (smaller)
    plot(xPoints, yPoints, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 3);
end

hold off;

%% Save the figure as PNG and TIFF
exportPNG = fullfile(workingDir, [filename, 'BrainMap.png']);
exportTIF = fullfile(workingDir, [filename, 'BrainMap.tif']);
print(gcf, exportPNG, '-dpng', '-r300');

% Capture the current axis and write the image as a TIFF file
frame = getframe(gca);
imwrite(frame.cdata, exportTIF);

fprintf('Brain map generated from CSV and saved as PNG and TIFF.\n');