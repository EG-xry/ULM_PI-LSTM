classdef ConvLSTMTracker < handle
    properties
        input_size
        net
        lstm_hidden_units = 64
    end
    
    methods
        function obj = ConvLSTMTracker(img_size)
            obj.input_size = [img_size(2), img_size(1), 1]; % [W, H, C]
            obj.build_network();
        end
        
        function build_network(obj)
            obj.net = build_compatible_network(obj.input_size);
        end
        
        function heatmap_seq = coords_to_heatmap(obj, coords_seq)
            num_frames = numel(coords_seq);
            heatmap_seq = zeros([obj.input_size(1:2), num_frames]);
            
            for t = 1:num_frames
                coords = coords_seq{t};
                heatmap = zeros(obj.input_size(1:2));
                
                if isempty(coords)
                    heatmap_seq(:,:,t) = heatmap;
                    continue;
                end
                
                % 坐标安全处理
                valid_x = coords(:,1) >= 1 & coords(:,1) <= obj.input_size(2);
                valid_y = coords(:,2) >= 1 & coords(:,2) <= obj.input_size(1);
                valid = valid_x & valid_y;
                coords = coords(valid, :);
                
                coords(:,1) = min(max(round(coords(:,1)), 1), obj.input_size(2));
                coords(:,2) = min(max(round(coords(:,2)), 1), obj.input_size(1));
                
                if ~isempty(coords)
                    linear_ind = sub2ind(obj.input_size(1:2), coords(:,2), coords(:,1));
                    heatmap(linear_ind) = 1;
                end
                
                heatmap_seq(:,:,t) = imgaussfilt(heatmap, 1.5);
            end
            heatmap_seq = reshape(heatmap_seq, [obj.input_size(1:2), 1, num_frames]);
        end
        
        function pred_tracks = track(obj, coords_seq)
            num_frames = numel(coords_seq);
            pred_tracks = cell(num_frames, 1);
            for t = 1:num_frames
                pred_tracks{t} = coords_seq{t}; % 占位符实现
            end
        end
    end
    
    methods (Static)
        function net = build_compatible_network()
            % 将网络构建函数移至类内部
            input_size = [128, 128, 1]; % 根据实际输入尺寸调整
            
            layers = [
                sequenceInputLayer(input_size, 'Name', 'input')
                convolution2dLayer(3, 32, 'Padding','same', 'Name', 'conv1')
                batchNormalizationLayer('Name', 'bn1')
                reluLayer('Name', 'relu1')
                lstmLayer(64, 'OutputMode','last', 'Name', 'lstm')
                fullyConnectedLayer(prod(input_size(1:2))*2)
                functionLayer(@(X) reshape(X, [input_size(1:2), 2]), 'Formattable', true)
                regressionLayer('Name', 'output')
            ];
            
            net = layerGraph(layers);
        end
    end
    
    
end