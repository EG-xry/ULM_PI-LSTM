%% PALA_InVivoULM_example.m : Post Processing - filtering, localization and tracking for in vivo data
% Simple script to perform ULM on in vivo data
% IQ are loaded, filtered and sent into ULM processing.
% Bubbles are detected, selected, then localized, and paired into tracks.
% The result is a list of interpolated tracks to reconstructed a full brain vascularization.
%
% Created by Arthur Chavignon 25/02/2020
% Improved and annotated by Eric Gao 20/02/2025
run('PALA_SetUpPaths.m')
%% Selected data file and saving folders
workingdir = [PALA_data_folder '/PALA_data_InVivoRatBrain'];
filename = 'PALA_InVivoRatBrain_';
cd(workingdir)

mydatapath = [workingdir filesep 'IQ' filesep filename];
workingdir = [workingdir '\Example'];mkdir(workingdir)

IQfiles = dir([mydatapath '*.mat']);
% Loads the first .mat file found in the director, specifically UF and Pdata
load([IQfiles(1).folder filesep IQfiles(1).name],'UF','PData');

%% Adapt parameters to your data
% A few parameters must be provided by the user depending of your input images (size of pixel, wavelength)
% These parameters will be copied and used later during the creation of the ULM structure.
load([IQfiles(1).folder filesep IQfiles(1).name],'IQ','UF');
% in this example, UF.TwFreq = 15MHz, UF.FrameRateUF = 1000Hz;

% Here you put the size of your data
SizeOfBloc = size(IQ);              % [nb_pixel_z,nb_pixel_x, nb_frame_per_bloc]

ScaleOfPixel = [1 1];               % [pixel_size_z, pixel_size_x]
% In this example, the size of pixels is lambda x lambda. The localization process will be
% performs in wavelength. For velocity rendering, velocities in [wavelength/s] will be converted in [mm/s].

% The imaging frame rate is required for velocity calculation, and temporal filtering.
framerate = UF.FrameRateUF;          % imaging framerate in [Hz]

% Number of blocs to process
Nbuffers = numel(IQfiles);          % number of bloc to process (used in the parfor)

% If pixel sizes are in wavelength, lambda must be provided for a velocity maps in mm/s,
lambda = 1540/(UF.TwFreq*1e3);      % in metrics like mm, um, cm... it will define the unity of the velocity (speed of sound/ frequency)
clear P UF IQ

%% Display and Format Input Data Information 
fprintf('=================================================\n');
fprintf('ULM Data Input Summary:\n');
fprintf('-------------------------------------------------\n');
fprintf('Working Directory: %s\n', workingdir);
fprintf('Data File Prefix: %s\n', filename);
fprintf('Number of IQ files detected: %d\n', numel(IQfiles));

% Load a sample data file to retrieve IQ and UF parameters if needed.
sampleFile = fullfile(IQfiles(1).folder, IQfiles(1).name);
% Load both IQ and UF variables explicitly
sampleData = load(sampleFile, 'IQ', 'UF');  

if isfield(sampleData, 'IQ')
    IQ_sample = sampleData.IQ;
    SizeOfBloc = size(IQ_sample);
    fprintf('IQ Data Dimensions: %d (z) x %d (x) x %d (frames per bloc)\n', ...
        SizeOfBloc(1), SizeOfBloc(2), SizeOfBloc(3));
else
    error('Sample file does not contain IQ data.');
end

if isfield(sampleData, 'UF')
    UF = sampleData.UF;
else
    error('Sample file does not contain UF data.');
end

fprintf('Pixel Scale (in current units): [%g, %g]\n', ScaleOfPixel(1), ScaleOfPixel(2));
fprintf('Imaging Frame Rate: %d Hz\n', framerate);

% Calculate the ultrasound wavelength lambda in mm (using 1540 m/s)
lambda = 1540/(UF.TwFreq*1e3);
fprintf('Ultrasound Wavelength (lambda): %.4f mm\n', lambda);

%% ULM parameters

% In this example, input pixel are isotropic and equal to lambda (pixelPitch_x = pixelPitch_y = lambda)
% All size defined later are expressed in lambda

res = 10; % final ratio of localization rendering, it's approximately resolution factor of localization in scale(1) units.
% for a pixel size of 100um, we can assume that ULM algorithm provides precision 10 (res) times
% smaller than pixel size. Final rendering will be reconstructed on a 10x10um grid.

% Precompute FWHM value to avoid parser confusion
fwhm_value = [1, 1] * 3;  % Results in [3, 3]

% 'numberOfParticles: Maximum number of microbubble detections (or particles) to consider per frame.

% 'size': The dimensions of the data block
   %   - SizeOfBloc(1): Number of pixels in the z (axial) direction.
   %   - SizeOfBloc(2): Number of pixels in the x (lateral) direction.
   %   - SizeOfBloc(3): Number of frames (time dimension) per data block.

% 'scale':Scaling factors for converting from pixel indices to physical units
   %   - ScaleOfPixel: [pixel_size_z, pixel_size_x] (e.g., in micrometers or wavelengths).
   %   - 1/framerate: Time scale, converting frame numbers to time (seconds per frame).

% 'res': The resolution factor: indicates that the localization precision is expected to be 'res' times better than the original pixel size.
% 'SVD_cutoff': Parameters for Singular Value Decomposition (SVD) filtering
   %   - The first value (5) specifies how many singular values to ignore (e.g., to remove clutter).
   %   - The second value (SizeOfBloc(3)) represents the total number of frames, setting the upper limit of the SVD window.

% 'max_linking_distance': The maximum distance (in scaled units) allowed between detections
   % when linking them across consecutive frames to form a continuous track.
   % This threshold helps to ensure that only spatially proximate localizations are linked.

% 'min_length': The minimum number of frames (or detections) required for a track to be considered valid.
% 'fwhm': Full Width at Half Maximum (FWHM) for the point spread function used during localization.
   % Precomputed as [3, 3], meaning the expected FWHM is 3 units in both the z and x directions.

% 'max_gap_closing': The maximum allowed gap (in frames) when linking localization detections.
   % A value of 0 means that no gap (i.e., missing detection) is tolerated between track points.

% 'interp_factor': The interpolation factor used when mapping localized points to the final super-resolved grid. 
   % Setting it to 1/res scales the coordinates appropriately,since a higher 'res' indicates a finer grid.


% 'LocMethod': Specifies the localization method to use.
   % In this case, 'Radial' indicates that the algorithm uses a radial symmetry-based
   % approach for pinpointing the microbubble positions.


ULM = struct('numberOfParticles', 90,...  % Number of particles per frame. (30-100)
    'size',[SizeOfBloc(1) SizeOfBloc(2) SizeOfBloc(3)],... 
    'scale',[ScaleOfPixel 1/framerate],...% Scale [z x dt], size of pixel in the scaling unit. (here, pixsize = 1*lambda)
    'res',res,...                       
    'SVD_cutoff',[5 SizeOfBloc(3)],...  % To be adapted to your clutter/SNR levels
    'max_linking_distance',2,...        % Maximum linking distance between two frames to reject pairing, in pixels units (UF.scale(1)). (2-4 pixel).
    'min_length', 15,...                % Minimum allowed length of the tracks in time. (5-20 frames)
    'fwhm',fwhm_value,...                  % Size [pixel] of the mask for localization. (3x3 for pixel at lambda, 5x5 at lambda/2). [fmwhz fmwhx]
    'max_gap_closing', 0,...            % Allowed gap in microbubbles' pairing. (if you want to skip frames 0)
    'interp_factor',1/res,...           % Interpfactor (decimation of tracks)
    'LocMethod','Radial'...             % Select localization algorithm (WA,Interp,Radial,CurveFitting,NoLocalization)
    );

ULM.ButterCuttofFreq = [50 249];        % Cutoff frequency (Hz) for additional filter. Typically [20 300] at 1kHz.
ULM.parameters.NLocalMax = 3;           % Safeguard on the number of maxLocal in the fwhm*fwhm grid (3 for fwhm=3, 7 for fwhm=5)
[but_b,but_a] = butter(2,ULM.ButterCuttofFreq/(framerate/2),'bandpass');
ULM.lambda = lambda;

ULM.velocity_tol = 1.0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now display the ULM structure.
if exist('ULM','var')
    fprintf('ULM Parameters:\n');
    disp(ULM);
else
    fprintf('ULM Parameters: (ULM structure not defined yet)\n');
end
fprintf('=================================================\n\n');

%% Visualization of Input Data
% Plot and save absolute IQ data from first frame
figure;
imagesc(abs(IQ_sample(:,:,1)));
colormap gray; colorbar;
axis image;
title('Absolute IQ Data - Frame 1');
xlabel('Pixel x');
ylabel('Pixel z');
% Export as PNG image
print(gcf, [workingdir filesep filename 'IQ_frame1'], '-dpng', '-r300');

% Plot and save histogram of IQ amplitudes
figure;
histogram(abs(IQ_sample(:)), 100);
title('Histogram of IQ Data Amplitudes');
xlabel('Amplitude');
ylabel('Frequency');
grid on;
% Export as PNG image
print(gcf, [workingdir filesep filename 'IQ_histogram'], '-dpng', '-r300');

%% Processing Parameters
Nbuffers = 12;
ProcessNumBlocks = min(999, Nbuffers); % Set to desired number of blocks (0 = all)
% Set to:
% - Specific number (e.g. 10) to process first 10 blocks
% - 0 to process all available blocks
% - min(N,X) to process up to X blocks while respecting available data
if ProcessNumBlocks <= 0 || ProcessNumBlocks > Nbuffers
    ProcessNumBlocks = Nbuffers; % Auto-correct to max available blocks
end
fprintf('\nProcessing %d/%d available blocks\n', ProcessNumBlocks, Nbuffers);

%% Load data, localize microbubbles, and store coordinate data
tic;
Track_tot = {};         % To store tracks per block
Coord_tot = {};         % To store localization coordinates in desired order [Intensity, X, Z, ImageIndex]
fprintf('--- ULM PROCESSING --- \n\n'); 
t1 = tic;

parfor hhh = 1:ProcessNumBlocks  % Now uses user-defined limit
    fprintf('Processing bloc %d/%d\n', hhh, ProcessNumBlocks);
    
    % Load IQ data from current block
    tmp = load([IQfiles(hhh).folder filesep IQfiles(hhh).name], 'IQ');
    
    % Filtering of IQ to remove clutter (optional)
    IQ_filt = SVDfilter(tmp.IQ, ULM.SVD_cutoff); 
    tmp = [];
    
    % Temporal filtering
    IQ_filt = filter(but_b, but_a, IQ_filt, [], 3);  %(optional)
    IQ_filt(~isfinite(IQ_filt)) = 0;
    
    % Detection and localization process (returns a list of coordinates in pixel)
    [MatTracking] = ULM_localization2D(abs(IQ_filt), ULM); 
    IQ_filt = [];
    
    % Convert pixels into isogrid (pixels are not necessarily isometric)
    MatTracking(:,2:3) = (MatTracking(:,2:3) - [1 1]) .* ULM.scale(1:2);
    
    % NEW: Reorder coordinate information to output as [Intensity, X, Z, ImageIndex]
    % Correct mapping based on raw MatTracking assumed as: [t, z, x, intensity]
    % So: Intensity -> column 4; X -> column 3; Z -> column 2; ImageIndex -> column 1.
    if size(MatTracking,2) < 4
        % If intensity is not provided, default to NaN.
        intensity = nan(size(MatTracking,1), 1);
        Coord = [intensity, MatTracking(:,3), MatTracking(:,2), MatTracking(:,1)];
    else
        Coord = [MatTracking(:,4), MatTracking(:,3), MatTracking(:,2), MatTracking(:,1)];
    end
    Coord_tot{hhh} = Coord;
    
    % Tracking algorithm (list of tracks)
    Track_tot_i = ULM_tracking2D(MatTracking, ULM, 'velocityinterp');
    
    % Save tracks for current block (optional: can also be saved individually)
    Track_tot{hhh} = Track_tot_i;
    
    % Clean-up variables for this iteration
    Track_tot_i = [];
    MatTracking = [];
end

Track_tot = cat(1, Track_tot{:});
Tend = toc(t1);
disp('Done');
fprintf('ULM done in %d hours %.1f minutes.\n', floor(Tend/3600), rem(Tend/60,60));

%% Output Localization Coordinates
% Concatenate all coordinate information collected into one matrix.
% Assume that each element of Coord_tot is in the order: [t, z, x, intensity]
Coord_all = cat(1, Coord_tot{:});

% Force reordering to the desired order: [Intensity, X, Z, ImageIndex]
% That is, [MatTracking(:,4), MatTracking(:,3), MatTracking(:,2), MatTracking(:,1)]
Coord_all = [Coord_all(:,4), Coord_all(:,3), Coord_all(:,2), Coord_all(:,1)];

% Option 1: Save coordinate information to a MAT file
save(fullfile(workingdir, [filename 'Coordinates.mat']), 'Coord_all');  

% Option 2: Export localization results to a CSV file 
exportLocalizationCSV = true;  % Set to false if CSV export is not needed
if exportLocalizationCSV
    % Create header row describing each column.
    header = {'Intensity', 'X', 'Z', 'ImageIndex'};
    % Combine header with data rows (converted to cells).
    csvData = [header; num2cell(Coord_all)];
    
    % Save CSV to the working directory.
    csvFilename = fullfile(workingdir, [filename 'Coordinates.csv']);
    writecell(csvData, csvFilename);
    fprintf('Localization results exported to CSV: %s\n', csvFilename);
    
    % Additional save path for CSV file on the Desktop.
    extraCsvPath = '/Users/eric/Desktop/ULM_PI-LSTM/PINN';
    if ~exist(extraCsvPath, 'dir')
        mkdir(extraCsvPath);
    end
    csvFilenameExtra = fullfile(extraCsvPath, [filename 'Coordinates.csv']);
    writecell(csvData, csvFilenameExtra);
    fprintf('Localization results also exported to CSV: %s\n', csvFilenameExtra);
end

% The previous printing block for the coordinates has been removed.

%% Create individual variable to save using v6 version.
% By cutting Tracks into different variables small than 2GB, the save v6 is faster than save v7.3
CutTracks = round(linspace(1,numel(Track_tot),4));
Track_tot_1 = Track_tot(CutTracks(1):CutTracks(2)-1);
Track_tot_2 = Track_tot(CutTracks(2):CutTracks(3)-1);
Track_tot_3 = Track_tot(CutTracks(3):end);
save([workingdir filesep filename 'example_tracks.mat'],'Track_tot_1','Track_tot_2','Track_tot_3','Tend','ULM','-v6')
clear Track_tot_1 Track_tot_2 Track_tot_3

% load([workingdir filesep filename 'example_tracks.mat'])
% Track_tot = cat(1,Track_tot_1,Track_tot_2,Track_tot_3);clear Track_tot_1 Track_tot_2 Track_tot_3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create MatOut %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% create the MatOut density with interpolated tracks for visual analysis, and with non interpolated tracks for aliasing index calculation.
% Define the size of SRpixel for displaying (default 10)
ULM.SRscale = ULM.scale(1)/ULM.res;
ULM.SRsize = round(ULM.size(1:2).*ULM.scale(1:2)/ULM.SRscale);

% Convert tracks into SRpixel
Track_matout = cellfun(@(x) (x(:,[1 2 3 4])+[1 1 0 0]*1)./[ULM.SRscale ULM.SRscale 1 1],Track_tot,'UniformOutput',0);
llz = [0:ULM.SRsize(1)]*ULM.SRscale;llx = [0:ULM.SRsize(2)]*ULM.SRscale;

%% Accumulate tracks on the final MatOut grid.
fprintf('--- CREATING MATOUTS --- \n\n')
MatOut = ULM_Track2MatOut(Track_matout,ULM.SRsize+[1 1]*1); %pos in superpix [z x]
MatOut_zdir = ULM_Track2MatOut(Track_matout,ULM.SRsize+[1 1]*1,'mode','2D_vel_z'); %pos in superpix [z x]
MatOut_vel = ULM_Track2MatOut(Track_matout,ULM.SRsize+[1 1]*1,'mode','2D_velnorm'); %pos in superpix [z x]
MatOut_vel = MatOut_vel*ULM.lambda; % Convert into [mm/s]

save([workingdir filesep filename 'example_matouts.mat'],'MatOut','MatOut_zdir','MatOut_vel','ULM','lambda','llx','llz')

%% Modified MatOut visualization section
close all;  % Close previous figures to avoid conflicts.
fig = figure('Visible','on');  % Explicitly create a new visible figure.
clf(fig);
set(fig, 'Position', [652 393 941 585]);  % Set position for consistent display.

IntPower = 1/3;SigmaGauss=0;
im=imagesc(llx,llz,MatOut.^IntPower);axis image
if SigmaGauss>0,im.CData = imgaussfilt(im.CData,SigmaGauss);end

title('ULM intensity display')
colormap(gca,gray(128))
clbar = colorbar;caxis(caxis*.8)  % add saturation in image
clbar.Label.String = 'number of counts';
clbar.TickLabels = round(clbar.Ticks.^(1/IntPower),1);xlabel('\lambda');ylabel('\lambda')
ca = gca;ca.Position = [.05 .05 .8 .9];
BarWidth = round(1./(ULM.SRscale*lambda)); % 1 mm
im.CData(size(MatOut,1)-50+[0:3],60+[0:BarWidth])=max(caxis);

drawnow;   % Force MATLAB to complete all pending drawing updates.
pause(0.1);  % Allow a short time for complete rendering.

% Ensure the figure handle is still valid. If not, recreate it.
if ~isgraphics(fig)
    warning('Figure handle lost. Recreating figure.');
    fig = figure('Visible','on');
    % Optionally, re-run your plotting commands here.
    % (For example, call a function that re-draws the figure.)
end

% Export the figure as a high-resolution PNG image.
exportFile = fullfile(workingdir, [filename, 'example_MatOut.png']);
print(fig, exportFile, '-dpng', '-r300');

%% MatOut intensity rendering, with axial direction color encoding
% Encodes the intensity with the Matout, but negative if the average velocity of the track is downward.
figure(2);clf,set(gcf,'Position',[652 393 941 585]);
velColormap = cat(1,flip(flip(hot(128),1),2),hot(128)); % custom velocity colormap
velColormap = velColormap(5:end-5,:); % remove white parts
IntPower = 1/4;
im=imagesc(llx,llz,(MatOut).^IntPower.*sign(imgaussfilt(MatOut_zdir,.8)));
im.CData = im.CData - sign(im.CData)/2;axis image
title(['ULM intensity display with axial flow direction'])
colormap(gca,velColormap)
caxis([-1 1]*max(caxis)*.7) % add saturation in image
clbar = colorbar;clbar.Label.String = 'Count intensity';
ca = gca;ca.Position = [.05 .05 .8 .9];
BarWidth = round(1./(ULM.SRscale*lambda)); % 1 mm
im.CData(size(MatOut,1)-50+[0:3],60+[0:BarWidth])=max(caxis);

print(gcf,[workingdir filesep filename 'example_MatOut_zdir'],'-dpng','-r750')
WriteTif(im.CData,ca.Colormap,[workingdir filesep filename 'example_MatOut_zdir.tif'],'caxis',caxis,'Overwrite',1)

%% Velocity magnitude rendering
% Encodes the norm velocity average in pixels
% vmax_disp = round(ULM.max_linking_distance*ULM.scale(1)*ULM.lambda/ULM.scale(3)*.6); % maximal displayed velocity, should be adapt to the imaged organ [mm/s]
vmax_disp  = ceil(quantile(MatOut_vel(abs(MatOut_vel)>0),.98)/10)*10;

figure(3);clf,set(gcf,'Position',[652 393 941 585]);
clbsize = [180,50];
Mvel_rgb = MatOut_vel/vmax_disp; % normalization
Mvel_rgb(1:clbsize(1),1:clbsize(2)) = repmat(linspace(1,0,clbsize(1))',1,clbsize(2)); % add velocity colorbar
Mvel_rgb = Mvel_rgb.^(1/1.5);Mvel_rgb(Mvel_rgb>1)=1;
Mvel_rgb = imgaussfilt(Mvel_rgb,.5);
Mvel_rgb = ind2rgb(round(Mvel_rgb*256),jet(256)); % convert ind into RGB

MatShadow = MatOut;MatShadow = MatShadow./max(MatShadow(:)*.3);MatShadow(MatShadow>1)=1;
MatShadow(1:clbsize(1),1:clbsize(2))=repmat(linspace(0,1,clbsize(2)),clbsize(1),1);
Mvel_rgb = Mvel_rgb.*(MatShadow.^IntPower);
Mvel_rgb = brighten(Mvel_rgb,.4);
BarWidth = round(1./(ULM.SRscale*lambda)); % 1 mm
Mvel_rgb(size(MatOut,1)-50+[0:3],60+[0:BarWidth],1:3)=1;
imshow(Mvel_rgb,'XData',llx,'YData',llz);axis on
title(['Velocity magnitude (0-' num2str(vmax_disp) 'mm/s)'])
ca = gca;ca.Position = [.05 .05 .8 .9];

print(gcf,[workingdir filesep filename 'example_VelMorm'],'-dpng','-r750')
imwrite(Mvel_rgb,[workingdir filesep filename 'example_VelMorm.tif'])

%% PowerDoppler rendering
% Comparison with a example of power Doppler rendering
% A few blocs of images are filtered with Singular Value Decomposition, and averaged to
% generate a PowerDoppler rendering. The aim of this section is only to get a comparison
% between PowerDoppler imaging and ULM process.
figure(4);clf,set(gcf,'Position',[652 393 941 585]);
PowDop = [];
for hhh=1:2:min(20,Nbuffers)
    tmp = load([IQfiles(hhh).folder filesep IQfiles(hhh).name],'IQ');
    IQ_filt = SVDfilter(tmp.IQ,ULM.SVD_cutoff);tmp = [];
    PowDop(:,:,end+1) = sqrt(sum(abs(IQ_filt).^2,3));
end
im=imagesc([0:ULM.size(2)-1].*ULM.scale(2),[0:ULM.size(1)-1].*ULM.scale(1),mean(PowDop,3).^(1/3));
axis image, colormap(gca,hot(128)),title(['Power Doppler'])
clbar = colorbar;
ca = gca;ca.Position = [.05 .05 .8 .9];caxis([10 max(im.CData(:))*.9])
BarWidth = round(1./(ULM.scale(2)*lambda)); % 1 mm
im.CData(size(im.CData,1)-2,3+[0:BarWidth])=max(caxis);

print(gcf,[workingdir filesep filename 'example_PowDop'],'-dpng','-r750')
WriteTif(im.CData,ca.Colormap,[workingdir filesep filename 'example_PowDop.tif'],'caxis',caxis,'Overwrite',1)

fprintf('PALA_InVivoULM_example.m completed.\n');