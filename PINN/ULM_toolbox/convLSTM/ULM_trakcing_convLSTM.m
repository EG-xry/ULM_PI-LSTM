classdef ULM_trakcing_convLSTM < handle
    properties
        % 网络参数
        input_size = [128, 128, 1]   % 输入热力图尺寸 [H,W,C]
        lstm_hidden_units = 64       % LSTM隐藏单元数
        net                          % 训练好的网络
    end
    
    methods
        %% 主入口函数
        function Tracks = track(obj, coords_seq)
            % 输入：coords_seq - 细胞数组，每个元素为N×2矩阵表示各帧坐标
            % 输出：Tracks - 细胞数组，每个元素为M×3矩阵[frame, x, y]
            
            % 生成热力图序列
            heatmap_seq = obj.coords_to_heatmap(coords_seq);
            
            % 模型预测位移场
            disp_fields = obj.predict_disp(heatmap_seq);
            
            % 生成最终轨迹
            Tracks = obj.generate_tracks(coords_seq, disp_fields);
        end
        
        %% 数据预处理：坐标转热力图
        function heatmap_seq = coords_to_heatmap(obj, coords_seq)
            num_frames = numel(coords_seq);
            heatmap_seq = zeros([obj.input_size, num_frames]);
            
            for t = 1:num_frames
                coords = coords_seq{t};
                heatmap = zeros(obj.input_size(1:2));
                
                % 坐标映射
                valid = coords(:,1)>=1 & coords(:,1)<=obj.input_size(2) & ...
                        coords(:,2)>=1 & coords(:,2)<=obj.input_size(1);
                coords = round(coords(valid, :));
                
                % 生成热力图
                for k = 1:size(coords,1)
                    x = coords(k,1);
                    y = coords(k,2);
                    heatmap(y,x) = 1;
                end
                
                % 高斯平滑
                heatmap_seq(:,:,t) = imgaussfilt(heatmap, 1.5);
            end
        end
        
        %% 传统算法生成伪标签（自动标注）
        function pseudo_tracks = generate_pseudo_labels(obj, coords_seq)
            % 使用最近邻算法生成初始轨迹
            max_distance = 5; % 像素单位
            pseudo_tracks = cell(1000,1);
            current_id = 1;
            
            for t = 1:numel(coords_seq)-1
                current_coords = coords_seq{t};
                next_coords = coords_seq{t+1};
                
                % 计算距离矩阵
                D = pdist2(current_coords, next_coords);
                
                % 最近邻匹配
                [~, idx] = min(D, [], 2);
                
                % 分配轨迹
                for k = 1:size(current_coords,1)
                    if D(k, idx(k)) < max_distance
                        if isempty(pseudo_tracks{current_id})
                            pseudo_tracks{current_id} = [t, current_coords(k,:)];
                        end
                        pseudo_tracks{current_id} = [pseudo_tracks{current_id}; 
                                                    t+1, next_coords(idx(k),:)];
                    else
                        current_id = current_id + 1;
                    end
                end
            end
            
            % 移除空轨迹
            pseudo_tracks = pseudo_tracks(~cellfun('isempty',pseudo_tracks));
        end
        
        %% ConvLSTM网络定义
        function build_network(obj)
            layers = [
                sequenceInputLayer([obj.input_size], 'Name', 'input')
                
                convolution2dLayer(3, 32, 'Padding','same', 'Name', 'conv1')
                batchNormalizationLayer('Name', 'bn1')
                reluLayer('Name', 'relu1')
                
                lstmLayer(obj.lstm_hidden_units, 'OutputMode','sequence', 'Name', 'lstm')
                
                convolution2dLayer(3, 2, 'Padding','same', 'Name', 'conv2')
                regressionLayer('Name', 'output')
            ];
            
            options = trainingOptions('adam', ...
                'MaxEpochs', 50, ...
                'MiniBatchSize', 8, ...
                'Plots', 'training-progress');
            
            obj.net = trainNetwork([], layers, options);
        end
        
        %% 位移场预测
        function disp_fields = predict_disp(obj, heatmap_seq)
            % 将热力图序列转换为网络输入格式
            input_data = reshape(heatmap_seq, [obj.input_size, 1, numel(heatmap_seq)]);
            
            % 预测位移场
            pred = predict(obj.net, input_data);
            
            % 解析输出
            disp_fields = squeeze(pred);
        end
        
        %% 轨迹生成后处理
        function Tracks = generate_tracks(obj, coords_seq, disp_fields)
            Tracks = cell(1000,1);
            current_id = 1;
            
            % 初始化首帧
            first_coords = coords_seq{1};
            for k = 1:size(first_coords,1)
                Tracks{current_id} = [1, first_coords(k,:)];
                current_id = current_id + 1;
            end
            
            % 逐帧更新
            for t = 1:size(disp_fields,3)-1
                current_disp = disp_fields(:,:,:,t);
                
                for track_id = 1:current_id-1
                    if size(Tracks{track_id},1) == t
                        last_pos = Tracks{track_id}(end,2:3);
                        
                        % 从位移场获取预测位移
                        dx = current_disp(round(last_pos(2)), round(last_pos(1)), 1);
                        dy = current_disp(round(last_pos(2)), round(last_pos(1)), 2);
                        
                        new_pos = last_pos + [dx, dy];
                        Tracks{track_id} = [Tracks{track_id}; t+1, new_pos];
                    end
                end
            end
            
            % 过滤短轨迹
            Tracks = Tracks(~cellfun('isempty',Tracks));
            track_lens = cellfun(@(x) size(x,1), Tracks);
            Tracks = Tracks(track_lens >= 5); % 保留至少5帧的轨迹
        end
    end
end