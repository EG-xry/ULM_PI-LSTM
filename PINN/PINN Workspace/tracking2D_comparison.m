function Tracks_out = ULM_tracking2D_csv(ULM)
    %% function Tracks_out = ULM_tracking2D_csv(ULM)
    % Takes bubble positions from a CSV file and returns a list of tracks.
    % The pairing of bubbles is based on partial assignment and minimization of the total 
    % distance using 'simpletracker' and a munkres implementation.
    % Reads from mb_ulm_4con.csv containing columns: Intensity, X, Y, ImageIndex
    % Outputs tracks to tracks_original.csv in format: track_id, point_index, x, y, t
    %
    % INPUTS:
    %       - ULM.max_linking_distance: maximal distance between two bubbles that can be paired
    %       - ULM.max_gap_closing: maximal gap of frames accepted to pair bubbles.
    %       - ULM.min_length: minimal track length (number of points)
    %       - ULM.res (optional): resolution (default is 1)
    %       - ULM.scale (optional): scale of data, [scale_z scale_x scale_t] (default: [1, 1, 1])
    %
    % OUTPUTS:
    %       - Tracks_out: a cell array in which each cell is a matrix with columns [y, x, frame]
    %
    % This version removes all interpolation and returns only the raw tracks.
    
    % If ULM is not provided, define default ULM parameters
    if nargin < 1 || isempty(ULM)
        disp('No ULM parameter provided. Using default ULM values.');
        ULM.max_linking_distance = 10;  % default linking distance
        ULM.max_gap_closing = 3;        % default gap closing parameter
        ULM.min_length = 5;             % default minimum track length (points)
        ULM.res = 1;                    % default resolution
        ULM.scale = [1, 1, 1];          % default scale [z, x, t]
    end

    %% Read CSV file
    disp('Reading CSV file mb_ulm_4con.csv...');
    data = readtable('mb_ulm_4con.csv');
    
    % Extract relevant columns (assuming columns are: Intensity, X, Y, ImageIndex)
    intensity = data.Intensity;
    x_pos = data.X;
    y_pos = data.Y;
    frame_num = data.ImageIndex;
    
    % Compute normalization parameters for x and y
    min_x = min(x_pos);
    max_x = max(x_pos);
    min_y = min(y_pos);
    max_y = max(y_pos);
    
    % Create MatTracking matrix [intensity, y position, x position, frame number]
    MatTracking = [intensity, y_pos, x_pos, frame_num];
    
    %% Define local parameters
    numberOfFrames = max(frame_num);
    ULM.size = [max(y_pos), max(x_pos), numberOfFrames];
    
    %% Renormalize frame numbers in case they don't start at 1
    minFrame = min(MatTracking(:,4));
    MatTracking(:,4) = MatTracking(:,4) - minFrame + 1;
    
    % Generate an array of indices for each frame
    index_frames = arrayfun(@(i) find(MatTracking(:,4) == i), 1:numberOfFrames, 'UniformOutput', false);
    
    % Group points by frame [y position, x position]
    Points = arrayfun(@(i) [MatTracking(index_frames{i},2), MatTracking(index_frames{i},3)], 1:numberOfFrames, 'UniformOutput', false);
    
    %% Tracking using simpletracker (raw tracking without interpolation)
    debug = false;
    [Simple_Tracks, Adjacency_Tracks] = simpletracker(Points, ...
        'MaxLinkingDistance', ULM.max_linking_distance, ...
        'MaxGapClosing', ULM.max_gap_closing, ...
        'Debug', debug);
    
    n_tracks = numel(Simple_Tracks);
    all_points = vertcat(Points{:});  
    
    % Construct raw tracks from the tracker output
    Tracks_raw = {};
    for i_track = 1:n_tracks
        track_id = Adjacency_Tracks{i_track};
        idFrame = MatTracking(track_id,4);
        track_points = cat(2, all_points(track_id,:), idFrame);
        if size(track_points,1) >= ULM.min_length
            Tracks_raw{end+1} = track_points;
        end
    end
    
    if isempty(Tracks_raw)
        disp(['Was not able to find tracks starting at frame ', num2str(minFrame)]);
        Tracks_out = {[0, 0, 0, 0]};
        return
    end
    
    %% Post processing of raw tracks (no interpolation)
    % Reformat each track so that columns 1, 2 and 3 correspond to y, x and frame respectively.
    Tracks_out = {};
    for k = 1:length(Tracks_raw)
        track_points = double(Tracks_raw{k});
        xi = track_points(:,2);
        zi = track_points(:,1);
        iFrame = track_points(:,3);
        
        % Retain tracks that meet the minimum length requirement (>= ULM.min_length)
        if length(zi) >= ULM.min_length
            Tracks_out{end+1} = [zi, xi, iFrame];
        end
    end
    
    % Remove any empty cells (if any)
    Tracks_out = Tracks_out(~cellfun('isempty',Tracks_out));
    
    %% Normalize the track coordinates to the [0,1] range
    disp('Normalizing track coordinates to [0, 1] range...');
    for k = 1:length(Tracks_out)
        track = Tracks_out{k};
        % Normalize y (first column) and x (second column) using the min and max computed above
        track(:,1) = (track(:,1) - min_y) / (max_y - min_y);
        track(:,2) = (track(:,2) - min_x) / (max_x - min_x);
        Tracks_out{k} = track;
    end
    
    %% Export tracks to CSV
    disp('Exporting tracks to tracks_normoriginal.csv...');
    
    % Prepare data for export
    track_id   = [];
    point_index = [];
    x_values   = [];
    y_values   = [];
    t_values   = [];
    
    for i_track = 1:length(Tracks_out)
        track = Tracks_out{i_track};
        num_points = size(track, 1);
        
        % For each point in the track, assign a track_id and point index.
        for i_point = 1:num_points
            track_id   = [track_id; i_track];
            point_index = [point_index; i_point];
            % Export the normalized coordinates:
            y_values   = [y_values; track(i_point, 1)];  % normalized y coordinate
            x_values   = [x_values; track(i_point, 2)];  % normalized x coordinate
            % If a frame number is available, use it; otherwise, use the point index.
            if size(track, 2) > 2
                t_values = [t_values; track(i_point, 3)];
            else
                t_values = [t_values; i_point];
            end
        end
    end
    
    % Create an output table and write to CSV
    output_table = table(track_id, point_index, x_values, y_values, t_values, ...
                         'VariableNames', {'track_id', 'point_index', 'x', 'y', 't'});
    writetable(output_table, 'tracks_normoriginal.csv');
    
    disp('Tracking complete. Results saved to tracks_normoriginal.csv');
    
    % If no output variable is requested, display the result in the Command Window.
    if nargout == 0
        disp('Tracks_out = ');
        disp(Tracks_out);
    end
end
    