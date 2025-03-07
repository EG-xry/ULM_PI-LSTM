%% PALA_InVivoULM_test.m : Post Processing - filtering, localization and tracking for in vivo data
% Simple script to perform ULM on in vivo data
% IQ are loaded, filtered and sent into ULM processing.
% Bubbles are detected, selected, then localized, and paired into tracks.
% The result is a list of interpolated tracks to reconstructed a full brain vascularization.
%
% Created by Arthur Chavignon 25/02/2020
% Refined and annotated by Eric Gao 20/02/2025
%
% AUTHORS: Arthur Chavignon, Baptiste Heiles, Vincent Hingot. CNRS, Sorbonne Universite, INSERM.
% Laboratoire d'Imagerie Biomedicale, Team PPM. 15 rue de l'Ecole de Medecine, 75006, Paris
% Code Available under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (see https://creativecommons.org/licenses/by-nc-sa/4.0/)
% ACADEMIC REFERENCES TO BE CITED
% Details of the code in the article by Heiles, Chavignon, Hingot, Lopez, Teston and Couture.  
% Performance benchmarking of microbubble-localization algorithms for ultrasound localization microscopy, Nature Biomedical Engineering, 2021.
% General description of super-resolution in: Couture et al., Ultrasound localization microscopy and super-resolution: A state of the art, IEEE UFFC 2018

run('PALA_SetUpPaths.m')

%% Selected data file and saving folders
workingdir = [PALA_data_folder '\PALA_data_InVivoRatBrain'];
filename = 'PALA_InVivoRatBrain_';
cd(workingdir)

mydatapath = [workingdir filesep 'IQ' filesep filename];
workingdir = [workingdir '\Example'];mkdir(workingdir)

IQfiles = dir([mydatapath '*.mat']);
load([IQfiles(1).folder filesep IQfiles(1).name],'UF','PData');

%% Adapt parameters to your data
% A few parameters must be provided by the user depending of your input images (size of pixel, wavelength)
% These parameters will be copied and used later during the creation of the ULM structure.
load([IQfiles(1).folder filesep IQfiles(1).name],'IQ','UF');

% in this example, UF.TwFreq = 15MHz, UF.FrameRateUF = 1000Hz;
% 超声波的波长为约 102.7 μm，lamda=c/f, c=1540 m/s, f=15*10^6
% 频率 1000 Hz 帧率表示每秒采集 1000 帧，可以准确跟踪 快速流动的微泡
% ULM.scale = [1 1 1 / UF.FrameRateUF]; 即，每隔 1 ms 采集一帧超声数据 
% 速度可以计算为 每帧就是1ms,而微泡的移动就是 移动距离，就可以计算出 移动速度


% Here you put the size of your data
SizeOfBloc = size(IQ);              % [nb_pixel_z,nb_pixel_x, nb_frame_per_bloc]

% 1个像素，是由1个波长形成的，所以 1 pixel = 1 lamda
% Here you put the size of pixel in the prefered unit. It can be um, mm, m, wavelength, or an arbitrary scale.
ScaleOfPixel = [1 1];               % [pixel_size_z, pixel_size_x]
% In that example, the size of pixels is lambda x lambda. The localization process will be
% performs in wavelength. For velocity rendering, velocities in [wavelength/s] will be converted in [mm/s].

% The imaging frame rate is required for velocity calculation, and temporal filtering.
framerate = UF.FrameRateUF;          % imaging framerate in [Hz]

% Number of blocs to process
Nbuffers = numel(IQfiles);          % number of bloc to process (used in the parfor)

% If pixel sizes are in wavelength, lambda must be provided for a velocity maps in mm/s,
% 人体软组织	≈ 1540 mm/s	医学超声成像（标准设定值）
lambda = 1540/(UF.TwFreq*1e3);      % in metrics like mm, um, cm... it will define the unity of the velocity (speed of sound/ frequency)


clear P UF IQ

%% Key Parameters for ULM Processing
% Wavelength calculation: lambda = c/f where:
% - c = 1540 m/s (speed of sound in soft tissue)
% - f = 15MHz (ultrasound frequency)
% Frame rate of 1000 Hz allows accurate tracking of fast-moving microbubbles
% ULM.scale = [1 1 1/framerate] means 1ms between frames
lambda = 1540/(UF.TwFreq*1e3);      % wavelength in mm

%% ULM Structure Configuration
% ULM structure contains critical parameters for processing:
% - 1 pixel = 1 wavelength for initial imaging
% - Resolution factor (res=10) enables lambda/10 precision in final rendering
% - SVD filtering removes first 5 singular values to reduce noise
% - max_linking_distance (2-4 pixels) determines maximum allowed bubble movement between frames
% - min_length filters out unstable tracks shorter than 15 frames
% - max_gap_closing=0 means no frame skipping allowed in tracking
% - FWHM=3 uses 3x3 window for bubble localization
ULM = struct('numberOfParticles', 90,...  % Max bubbles per frame
    'size',[SizeOfBloc(1) SizeOfBloc(2) SizeOfBloc(3)],...
    'scale',[ScaleOfPixel 1/framerate],...
    'res',res,...
    'SVD_cutoff',[5 SizeOfBloc(3)],...
    'max_linking_distance',2,...
    'min_length', 15,...
    'fwhm',[1 1]*3,...
    'max_gap_closing', 0,...
    'interp_factor',1/res,...
    'LocMethod','Radial'...
    );

% Butterworth filter parameters:
% - Removes low-frequency tissue noise (<50 Hz)
% - Removes high-frequency electronic noise (>249 Hz)
% - Improves signal quality for ULM processing
ULM.ButterCuttofFreq = [50 249];

% Safety parameter: Maximum number of local maxima in FWHM window
% Too many local maxima likely indicates noise rather than individual bubbles
ULM.parameters.NLocalMax = 3;

%% Processing Pipeline
% 1. Load IQ data block
% 2. Apply SVD filtering to remove clutter
% 3. Apply temporal Butterworth filtering
% 4. Perform bubble detection and localization
% 5. Convert pixel coordinates to physical coordinates
% 6. Generate tracks from detected bubbles
for hhh = 1:nIQblock
    fprintf('Processing bloc %d/%d\n',hhh,nIQblock);
    
    % Load IQ data (or other kind of images without compression)
    tmp = load([IQfiles(hhh).folder filesep IQfiles(hhh).name],'IQ');

    % Filtering of IQ to remove clutter (optional)
    % 过滤 IQ 以消除杂波（可选）
    IQ_filt = SVDfilter(tmp.IQ,ULM.SVD_cutoff);
    tmp = [];

    % Temporal filtering
    % 去除噪声：减少高频噪声（如设备抖动）或低频漂移（如探头运动）。
    IQ_filt = filter(but_b,but_a,IQ_filt,[],3); %(optional) % 用到上面设置的好的ButterWorth通道
    IQ_filt(~isfinite(IQ_filt))=0;

    % Detection and localization process (return a list of coordinates in pixel)
    % 最重要的函数和处理

    [MatTracking] = ULM_localization2D(abs(IQ_filt),ULM); 
    IQ_filt=[];

    % MatTracking 是最重要的数据，结果数据。

    % Convert pixel into isogrid (pixel are not necessary isometric);
    % 将像素转换为等距网格（像素不必是等距的）；
    % 该代码将 MatTracking 从像素坐标转换为物理坐标
    % 作用将 (1,1) 变为 (0,0)，对齐坐标原点
    MatTracking(:,2:3) = (MatTracking(:,2:3) - [1 1]).*ULM.scale(1:2);


    % 复制原始像素坐标，这里把像素文件保存下来
    MatTracking_Pixel = MatTracking(:,2:3); % 保存原始像素坐标



    % Tracking algorithm (list of tracks)
    % 生成轨迹，重要。也是本次论文要改进的地方。



    % ULM_tracking2D是原来github上的源文件
    Track_tot_i = ULM_tracking2D(MatTracking,ULM);


    % 保存当前数据块的轨迹和原始像素信息
    save([workingdir filesep filename '_tracks' num2str(hhh,'%.3d') '.mat'], ...
        'Track_tot_i', 'MatTracking', 'MatTracking_Pixel', 'ULM', '-v7.3');


    % ULM_tracking2D_Enhanced.m是deepseek写的改进后的算法，测试中，尚未通过
    % Track_tot_i = ULM_tracking2D_Enhanced(MatTracking,ULM);

    % Chatgpt 建议写的代码
    disp(Track_tot_i);

    % 绘制轨迹
    figure; hold on;
    for i = 1:length(Track_tot_i)
        plot(Track_tot_i{i}(:,2), Track_tot_i{i}(:,1), '-');
    end
    xlabel('X 位置');
    ylabel('Z 位置');
    title('ULM 轨迹');
    grid on;


    % Saving part:
    %--- if for, you can save tracks at each loop to avoid RAM out of memory
    % save([workingdir filesep filename '_tracks' num2str(hhh,'%.3d') '.mat'],'Track_tot_i','ULM') %
    %--- if parfor you can cat all tracks in a huge cells variable
    Track_tot{hhh} = Track_tot_i;
    
    Track_tot_i={};

    MatTracking = [];
end


% 开始从这里注释
%{
parfor hhh = 1:min(999,Nbuffers) % can be used with parallel pool
% for hhh = 1:Nbuffers
    fprintf('Processing bloc %d/%d\n',hhh,Nbuffers);
    % Load IQ data (or other kind of images without compression)
    tmp = load([IQfiles(hhh).folder filesep IQfiles(hhh).name],'IQ');

    % Filtering of IQ to remove clutter (optional)
    % 过滤 IQ 以消除杂波（可选）
    IQ_filt = SVDfilter(tmp.IQ,ULM.SVD_cutoff);tmp = [];

    % Temporal filtering
    IQ_filt = filter(but_b,but_a,IQ_filt,[],3); %(optional)
    IQ_filt(~isfinite(IQ_filt))=0;

    % Detection and localization process (return a list of coordinates in pixel)
    % 最重要的函数和处理

    [MatTracking] = ULM_localization2D(abs(IQ_filt),ULM); IQ_filt=[];

    % Convert pixel into isogrid (pixel are not necessary isometric);
    % 将像素转换为等距网格（像素不必是等距的）；
    MatTracking(:,2:3) = (MatTracking(:,2:3) - [1 1]).*ULM.scale(1:2);

    % Tracking algorithm (list of tracks)
    % 生成轨迹，重要。也是本次论文要改进的地方。

    Track_tot_i = ULM_tracking2D(MatTracking,ULM);

    % Saving part:
    %--- if for, you can save tracks at each loop to avoid RAM out of memory
    % save([workingdir filesep filename '_tracks' num2str(hhh,'%.3d') '.mat'],'Track_tot_i','ULM') %
    %--- if parfor you can cat all tracks in a huge cells variable
    Track_tot{hhh} = Track_tot_i;
    Track_tot_i={};MatTracking = [];
end

Track_tot = cat(1,Track_tot{:});
Tend = toc(t1);disp('Done')
fprintf('ULM done in %d hours %.1f minutes.\n', floor(Tend/60/60), rem(Tend/60,60));

%% Create individual variable to save using v6 version.
% By cutting Tracks into different variables small than 2GB, the save v6 is faster than save v7.3
% 为了兼容v6文件格式，把结果文件切成2GB大小的

CutTracks = round(linspace(1,numel(Track_tot),4));
Track_tot_1 = Track_tot(CutTracks(1):CutTracks(2)-1);
Track_tot_2 = Track_tot(CutTracks(2):CutTracks(3)-1);
Track_tot_3 = Track_tot(CutTracks(3):end);

save([workingdir filesep filename 'example_tracks.mat'],'Track_tot_1','Track_tot_2','Track_tot_3','Tend','ULM','-v6')
% example_tracks 结果文件，很大，3.4Gb

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

%% MatOut intensity rendering, with compression factor
% 第一张图，灰度图
fprintf('--- GENERATING IMAGE RENDERINGS --- \n\n')
figure(1);clf,set(gcf,'Position',[652 393 941 585]);
IntPower = 1/3;SigmaGauss=0;

% 该代码用于 渲染 ULM 超分辨率影像 (MatOut)，并进行 Gamma 变换 (IntPower) 以增强对比度。
% 增强低信号区域，防止高信号过曝
% 该代码是 ULM 超分辨率影像 MatOut 可视化的核心，适用于微血管 5-10 μm 级别超分辨率血流分析
im=imagesc(llx,llz,MatOut.^IntPower);

axis image
if SigmaGauss>0,im.CData = imgaussfilt(im.CData,SigmaGauss);
end

title('ULM intensity display')
colormap(gca,gray(128))
clbar = colorbar;caxis(caxis*.8)  % add saturation in image
clbar.Label.String = 'number of counts';
clbar.TickLabels = round(clbar.Ticks.^(1/IntPower),1);xlabel('\lambda');ylabel('\lambda')
ca = gca;ca.Position = [.05 .05 .8 .9];
BarWidth = round(1./(ULM.SRscale*lambda)); % 1 mm

% 在 MatOut 影像底部添加 1mm 标尺
% 标尺高度 4 像素，长度 1mm 对应像素数
% 使用 max(caxis()) 设定亮度，确保标尺清晰可见
im.CData(size(MatOut,1)-50+[0:3],60+[0:BarWidth])=max(caxis);

% 将当前 figure 保存为 PNG 格式，分辨率750，Matout文件
print(gcf,[workingdir filesep filename 'example_MatOut'],'-dpng','-r750')

% 第一张图
WriteTif(im.CData,ca.Colormap,[workingdir filesep filename 'example_MatOut.tif'],'caxis',caxis,'Overwrite',1)

%% MatOut intensity rendering, with axial direction color encoding
% Encodes the intensity with the Matout, but negative if the average velocity of the track is downward.
% 第2张图，带有轴向的彩色的图

figure(2);clf,set(gcf,'Position',[652 393 941 585]);
velColormap = cat(1,flip(flip(hot(128),1),2),hot(128)); % custom velocity colormap
velColormap = velColormap(5:end-5,:); % remove white parts
IntPower = 1/4;
% 下一步产生图形
im=imagesc(llx,llz,(MatOut).^IntPower.*sign(imgaussfilt(MatOut_zdir,.8)));
im.CData = im.CData - sign(im.CData)/2;axis image
title(['ULM intensity display with axial flow direction'])
colormap(gca,velColormap)
% 下一步产生彩色
caxis([-1 1]*max(caxis)*.7) % add saturation in image
clbar = colorbar;clbar.Label.String = 'Count intensity';
ca = gca;ca.Position = [.05 .05 .8 .9];
BarWidth = round(1./(ULM.SRscale*lambda)); % 1 mm
im.CData(size(MatOut,1)-50+[0:3],60+[0:BarWidth])=max(caxis);

% 第2张图，带有轴向的彩色的图，z方向（向身体内或是向外）
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
% 下一步产生图形
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

%}

fprintf('PALA_InVivoULM_test.m completed.\n');



run('PALA_SetUpPaths.m')

%% Selected data file and saving folders
workingdir = [PALA_data_folder '\PALA_data_InVivoRatBrain'];
filename = 'PALA_InVivoRatBrain_';
cd(workingdir)

mydatapath = [workingdir filesep 'IQ' filesep filename];
workingdir = [workingdir '\Example']; mkdir(workingdir)

IQfiles = dir([mydatapath '*.mat']);
load([IQfiles(1).folder filesep IQfiles(1).name],'UF','PData');

%% Adapt parameters to your data
load([IQfiles(1).folder filesep IQfiles(1).name],'IQ','UF');

SizeOfBloc = size(IQ);              % [nb_pixel_z,nb_pixel_x, nb_frame_per_bloc]
ScaleOfPixel = [1 1];               % [pixel_size_z, pixel_size_x]
framerate = UF.FrameRateUF;          % imaging framerate in [Hz]
Nbuffers = numel(IQfiles);          % number of blocs to process
lambda = 1540/(UF.TwFreq*1e3);      % wavelength (e.g., in mm if using mm/s)

clear P UF IQ

%% ULM parameters
res = 10; % resolution factor
ULM = struct('numberOfParticles', 90,...
    'size',[SizeOfBloc(1) SizeOfBloc(2) SizeOfBloc(3)],...
    'scale',[ScaleOfPixel 1/framerate],...
    'res',res,...
    'SVD_cutoff',[5 SizeOfBloc(3)],...
    'max_linking_distance',2,...
    'min_length', 15,...
    'fwhm',[1 1]*3,...
    'max_gap_closing', 0,...
    'interp_factor',1/res,...
    'LocMethod','Radial'...
    );
ULM.ButterCuttofFreq = [50 249];
ULM.parameters.NLocalMax = 3;
[but_b,but_a] = butter(2,ULM.ButterCuttofFreq/(framerate/2),'bandpass');
ULM.lambda = lambda;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data and localize microbubbles %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear IQ Track_tot Track_tot_interp ProcessingTime
tic; Track_tot = {};
fprintf('--- ULM PROCESSING --- \n\n'); t1=tic;

fprintf('--- Begin Deal IQ Files  --- \n\n');

% Process a subset of blocks (here, nIQblock blocks)
nIQblock = 1;

for hhh = 1:nIQblock
    fprintf('Processing bloc %d/%d\n', hhh, nIQblock);
    
    % Load IQ data for current bloc
    tmp = load([IQfiles(hhh).folder filesep IQfiles(hhh).name],'IQ');
    
    % Filtering of IQ to remove clutter (optional)
    IQ_filt = SVDfilter(tmp.IQ,ULM.SVD_cutoff);
    tmp = [];
    
    % Temporal filtering with Butterworth filter
    IQ_filt = filter(but_b, but_a, IQ_filt, [], 3);
    IQ_filt(~isfinite(IQ_filt)) = 0;
    
    % Detection and localization process (returns list of coordinates)
    [MatTracking] = ULM_localization2D(abs(IQ_filt), ULM); 
    IQ_filt = [];
    
    % Convert pixel coordinates into physical units (origin shifted)
    MatTracking(:,2:3) = (MatTracking(:,2:3) - [1 1]) .* ULM.scale(1:2);
    
    % Tracking algorithm to generate tracks (cell array of tracks)
    Track_tot_i = ULM_tracking2D(MatTracking, ULM);
    
    % (Optional: Display the tracks for the current bloc)
    figure; hold on;
    for i = 1:length(Track_tot_i)
        plot(Track_tot_i{i}(:,2), Track_tot_i{i}(:,1), '-');
    end
    xlabel('X 位置');
    ylabel('Z 位置');
    title(sprintf('ULM 轨迹 - Bloc %d', hhh));
    grid on;
    
    % Save current bloc tracks into the cumulative cell array
    Track_tot{hhh} = Track_tot_i;
    
    Track_tot_i = {};
    MatTracking = [];
end

%% NEW: Cumulative overlay of tracks from all processed IQ blocks
% This section creates a single figure overlaying the tracks from all blocks.
% To help avoid excessive clutter you can change "plotSkip" to only plot every nth track.
plotSkip = 1;  % Set to a value >1 (e.g., 2 or 3) to plot only every nth track if needed

figure; hold on;
for h = 1:length(Track_tot)
    tracks_block = Track_tot{h};
    for j = 1:plotSkip:length(tracks_block)
         % Only plot tracks that are longer than the minimum allowed length
         if size(tracks_block{j},1) >= ULM.min_length
             % Plot using a light blue color and a modest line width.
             % (Adjust the 'Color' vector or 'LineWidth' if desired.)
             plot(tracks_block{j}(:,2), tracks_block{j}(:,1), '-', 'LineWidth', 1, 'Color', [0 0 1]);
         end
    end
end
xlabel('X 位置');
ylabel('Z 位置');
title('累积 ULM 轨迹 (Cumulative Track Overlay)');
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (The rest of the original code continues below)
% [The rest of your original code is left unchanged, including the MatOut creation,
% rendering, and saving sections.]
%
% (For example, the sections that create MatOut, display intensity images,
% power Doppler, etc. remain as originally written.)
%
% fprintf('PALA_InVivoULM_test.m completed.\n');

% 添加ULM结构体验证
if ~isstruct(ULM)
    error('ULM必须是结构体');
end

% 确保所有数值字段使用正确类型
required_fields = {'scale', 'numberOfFrames', 'SVD_cutoff'};
for i = 1:length(required_fields)
    if isfield(ULM, required_fields{i})
        ULM.(required_fields{i}) = double(ULM.(required_fields{i}));
    end
end

% 检查scale字段的维度
if numel(ULM.scale) < 2
    ULM.scale = [ULM.scale, ULM.scale]; % 保证至少2个维度
end

% 初始化时确保参数类型正确
ULM.scale = double(ULM.scale);
ULM.numberOfFrames = double(ULM.numberOfFrames);
ULM.SVD_cutoff = double(ULM.SVD_cutoff);

