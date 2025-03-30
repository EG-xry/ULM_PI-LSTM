% FinalImage.m : Generate final images from track data stored in tracks.csv
% This script loads a CSV file containing tracks in the format:
%    track_id, point_index, x, z, t
% It then groups the track points by track_id, computes approximate velocities using 
% finite differences, and then creates three graphics:
%   (1) Intensity (density) display,
%   (2) Intensity display with axial flow direction (diverging color scale),
%   (3) Velocity magnitude display.
%
% In each case, we "rotate" the image (i.e. apply a 90° clockwise transformation) 
% so that the final display has a horizontal direction from about 0 to 120 λ and 
% a vertical direction from 0 to 80 λ.
%
% The custom colormap for velocity magnitude goes from purplish blue ([51 13 170]/255)
% at zero up to yellow ([246 226 0]/255) at maximum. The axial flow image uses a diverging 
% colormap based on the original code:
%   velColormap = cat(1, flip(flip(hot(128),1),2), hot(128));
%   velColormap = velColormap(5:end-5,:);
%
% (Caveat: If some graphs cannot be reproduced due to missing data, simply ignore those.)

clear; close all; clc;

%% 1. Load track data from CSV
% CSV assumed to have header: track_id, point_index, x, z, t
data = readtable('tracks.csv');

% Group rows by track_id
track_ids = unique(data.track_id);
numTracks = length(track_ids);
Track_tot = cell(numTracks,1);

for i = 1:numTracks
    % Extract data for the current track, sort by point_index
    idx = (data.track_id == track_ids(i));
    trackData = data(idx,:);
    trackData = sortrows(trackData, 'point_index');
    
    % As the CSV is [track_id, point_index, x, z, t] but we desire the image
    % such that the horizontal axis is x (0 to 120) and vertical is z (0 to 80),
    % we swap the order so that the track matrix is [z, x, ...]
    z = trackData.z;
    x = trackData.x;
    t = trackData.t;
    
    % Compute approximate velocities using finite differences
    if height(trackData) >= 2
        dt = diff(t);
        % Avoid division by zero if timestamps coincide.
        v_z = diff(z) ./ dt;
        v_x = diff(x) ./ dt;
        % Duplicate last computed velocity for the last point:
        v_z = [v_z; v_z(end)];
        v_x = [v_x; v_x(end)];
    else
        v_z = 0;
        v_x = 0;
    end
    
    % Build track matrix with:
    % Column 1: z (axial coordinate)
    % Column 2: x (lateral coordinate)
    % Column 3: v_z (axial velocity)
    % Column 4: v_x (lateral velocity)
    Track_tot{i} = [z, x, v_z, v_x];
end

%% 2. Determine image grid and shift tracks
% Compute bounding box using (z,x) coordinates
all_z = []; all_x = [];
for i = 1:numTracks
    all_z = [all_z; Track_tot{i}(:,1)];
    all_x = [all_x; Track_tot{i}(:,2)];
end
min_z = min(all_z);
min_x = min(all_x);
max_z = max(all_z);
max_x = max(all_x);

% Shift tracks so that coordinates are positive (and start near 1)
offset_z = floor(min_z)-1;
offset_x = floor(min_x)-1;
for i = 1:numTracks
    Track_tot{i}(:,1:2) = Track_tot{i}(:,1:2) - [offset_z, offset_x];
end

% Define original image size (in pixels)
img_rows = ceil(max_z - offset_z) + 10;  % vertical size
img_cols = ceil(max_x - offset_x) + 10;    % horizontal size

%% 3. Define ULM parameters for rendering
% We assume input pixel size = 1 (in arbitrary units)
res = 10;  % resolution factor (e.g., a factor of 10 improvement)

ULM.scale = [1, 1, 1];     % [pixel_size_z, pixel_size_x, 1]
ULM.res = res;
ULM.size = [img_rows, img_cols, 1];  % spatial dimensions of the original grid
ULM.SRscale = ULM.scale(1) / ULM.res;  % super-resolution scale (1/res)
ULM.SRsize = round(ULM.size(1:2) .* ULM.scale(1:2) / ULM.SRscale);

%% 4. Transform tracks for ULM_Track2MatOut
% Map the tracks onto the super-resolved grid.
% The transformation is:
%    (track(:,[1 2 3 4]) + [1 1 0 0]) ./ [ULM.SRscale ULM.SRscale 1 1]
Track_matout = cellfun(@(x) (x(:,[1,2,3,4]) + repmat([1,1,0,0], size(x,1), 1)) ...
    ./ repmat([ULM.SRscale, ULM.SRscale, 1, 1], size(x,1), 1), Track_tot, 'UniformOutput', false);

% Although ULM_Track2MatOut produces images according to ULM.SRsize,
% for display we wish to force the physical axis limits to be:
% horizontal: 0 to 120 λ, vertical: 0 to 80 λ.
x_disp = [0, 120];  % horizontal (λ)
z_disp = [0, 80];   % vertical (λ)

%% 5. Generate Final Images using ULM_Track2MatOut
IntPower = 1/3;
SigmaGauss = 0;  % No smoothing (set >0 if desired)

% ---- Figure 1: Intensity (density) display ----
MatOut = ULM_Track2MatOut(Track_matout, ULM.SRsize + [1,1]);
% Rotate the image by 90° clockwise (i.e. -90° using rot90)
rotIntImage = rot90(MatOut.^IntPower, -1);

fig1 = figure;
% Use fixed axis limits so that horizontal spans 0 to 120 and vertical 0 to 80
imagesc(x_disp, z_disp, rotIntImage);
axis image;
colormap(gray(128));
colorbar;
title('ULM intensity display');
xlabel('\lambda'); ylabel('\lambda');
print(fig1, 'FinalImage_Intensity.png', '-dpng', '-r300');

% ---- Figure 2: Intensity display with axial flow direction ----
% Here the tracks are processed in '2D_vel_z' mode to encode the sign of axial velocity.
MatOut_zdir = ULM_Track2MatOut(Track_matout, ULM.SRsize + [1,1], 'mode', '2D_vel_z');
% Compute the intensity image modulated by the sign of filtered axial velocity.
rawAxial = (MatOut).^IntPower .* sign(imgaussfilt(MatOut_zdir, 0.8));
% Subtract half the sign to shift the scale as in the original code.
rawAxial = rawAxial - sign(rawAxial)/2;
% Rotate the axial flow image so that horizontal is 0-120 λ and vertical 0-80 λ.
rotAxial = rot90(rawAxial, -1);

fig2 = figure;
% Use the same display limits as for the intensity map.
imagesc(x_disp, z_disp, rotAxial);
axis image;
% Create the diverging colormap as in the original code:
% Concatenate a flipped version of hot(128) with hot(128), then remove extreme values.
velColormap = cat(1, flip(flip(hot(128),1),2), hot(128)); 
velColormap = velColormap(5:end-5,:);
colormap(gca, velColormap);
% Set color-axis saturation.
caxis([-1 1]*max(caxis)*0.7);
clbar = colorbar;
clbar.Label.String = 'Count intensity';
% Optionally adjust the axes position.
ca = gca; ca.Position = [.05 .05 .8 .9];

% (Optional) Draw a scale bar. Here we assume lambda = 1 (adjust if needed).
lambda = 1;  
BarWidth = round(1/(ULM.SRscale*lambda)); % e.g., ~1 unit wide
% Insert a white bar into the rotated axial image.
rotAxial(end-50+[0:3], 60+[0:BarWidth]) = max(caxis);
print(fig2, 'FinalImage_Intensity_Axial.png', '-dpng', '-r300');

% If you have the function WriteTif, you can also export as a TIFF.
if exist('WriteTif', 'file')
    WriteTif(rotAxial, ca.Colormap, 'FinalImage_Intensity_Axial.tif', 'caxis', caxis, 'Overwrite', 1);
end

% ---- Figure 3: Velocity magnitude rendering ----
% Here we process the tracks in '2D_velnorm' mode.
MatOut_vel = ULM_Track2MatOut(Track_matout, ULM.SRsize + [1,1], 'mode', '2D_velnorm');

% Estimate a maximum displayed velocity using the 98th percentile.
nonzero_vel = MatOut_vel(abs(MatOut_vel) > 0);
if ~isempty(nonzero_vel)
    vmax_disp = ceil(quantile(nonzero_vel, 0.98) / 10) * 10;
else
    vmax_disp = 1;
end

% Normalize and process the velocity magnitude image
Mvel_rgb = MatOut_vel / vmax_disp;  % normalized
Mvel_rgb = Mvel_rgb.^(1/1.5);
Mvel_rgb(Mvel_rgb > 1) = 1;
Mvel_rgb = imgaussfilt(Mvel_rgb, 0.5);
Mvel_rgb_ind = round(Mvel_rgb * 256); 
Mvel_rgb_ind(Mvel_rgb_ind > 256) = 256;

% Build a custom colormap for velocity magnitude:
% from purplish blue ([51,13,170]) at zero to yellow ([246,226,0]) at maximum.
nColors = 256;
customVMCM = [linspace(51/255, 246/255, nColors)', linspace(13/255, 226/255, nColors)', linspace(170/255, 0, nColors)'];
Mvel_rgb_rgb = ind2rgb(Mvel_rgb_ind, customVMCM);
% Rotate the velocity magnitude RGB image
rotVelRGB = rot90(Mvel_rgb_rgb, -1);

fig3 = figure;
% Display with desired axis extents.
imshow(rotVelRGB, 'XData', x_disp, 'YData', z_disp);
axis on;
title(sprintf('Velocity magnitude (0-%d)', vmax_disp));
colorbar;
print(fig3, 'FinalImage_Velocity.png', '-dpng', '-r300');

%% End of FinalImage.m
