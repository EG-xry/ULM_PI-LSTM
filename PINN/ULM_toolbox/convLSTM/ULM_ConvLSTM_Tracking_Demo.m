%% 主程序：ConvLSTM微泡跟踪全流程演示
clear; clc; close all;

%% 第一部分：生成模拟数据
rng(2023); % 固定随机种子
num_frames = 100;  % 总帧数
num_particles = 50; % 每帧微泡数量
img_size = [128, 128]; % 图像尺寸

% 生成连续运动轨迹
fprintf('生成模拟数据...\n');
[coords_seq, true_tracks] = generate_simulated_data(img_size, num_frames, num_particles);

%% 第二部分：初始化跟踪器
tracker = ConvLSTMTracker(img_size);

%% 第三部分：生成训练数据
fprintf('\n准备训练数据...\n');
% 生成热力图序列
heatmap_seq = tracker.coords_to_heatmap(coords_seq);

% 生成训练序列
seq_length = 5;  % 输入序列长度
[X_train, Y_train] = prepare_sequence_data(heatmap_seq, seq_length);

%% 第四部分：训练ConvLSTM模型
fprintf('\n训练模型中...\n');

% 构建网络
tracker.net = build_compatible_network(tracker.input_size);

% 配置训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 8, ...
    'InitialLearnRate', 1e-4, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

% 执行训练
tracker.net = trainNetwork(X_train, Y_train, tracker.net.Layers, options);

%% 第五部分：在新数据上测试
fprintf('\n运行跟踪...\n');
test_coords = coords_seq(1:50); % 使用前50帧作为测试
pred_tracks = tracker.track(test_coords);

%% 第六部分：结果可视化
fprintf('\n可视化结果...\n');
visualize_results(true_tracks, pred_tracks, img_size);

%% 辅助函数定义
function [coords_seq, true_tracks] = generate_simulated_data(img_size, num_frames, num_particles)
    true_tracks = cell(num_particles, 1);
    coords_seq = cell(num_frames, 1);
    
    % 初始化随机位置
    start_pos = rand(num_particles, 2) .* img_size;
    velocities = randn(num_particles, 2) * 0.8;
    
    for p = 1:num_particles
        true_tracks{p} = zeros(num_frames, 3);
        true_tracks{p}(1,:) = [1, start_pos(p,:)];
    end
    
    for t = 2:num_frames
        current_coords = zeros(num_particles, 2);
        for p = 1:num_particles
            velocities(p,:) = velocities(p,:) + randn(1,2)*0.2;
            new_pos = true_tracks{p}(t-1,2:3) + velocities(p,:);
            new_pos = max(new_pos, 1);
            new_pos = min(new_pos, img_size);
            
            true_tracks{p}(t,:) = [t, new_pos];
            current_coords(p,:) = new_pos;
        end
        coords_seq{t} = current_coords + randn(size(current_coords))*0.5;
    end
end

function [X_seq, Y_seq] = prepare_sequence_data(heatmap_seq, seq_length)
    num_frames = size(heatmap_seq,4);
    num_sequences = num_frames - seq_length;
    
    X_seq = cell(num_sequences,1);
    Y_seq = cell(num_sequences,1);
    
    for i = 1:num_sequences
        X_seq{i} = heatmap_seq(:,:,:,i:i+seq_length-1);
        Y_seq{i} = heatmap_seq(:,:,:,i+seq_length) - heatmap_seq(:,:,:,i+seq_length-1);
    end
end

function visualize_results(true_tracks, pred_tracks, img_size)
    figure('Position', [100 100 1200 500])
    
    subplot(1,2,1); hold on;
    for i = 1:min(50, length(true_tracks))
        track = true_tracks{i};
        plot(track(:,2), track(:,3), 'LineWidth', 1.2)
    end
    title('真实轨迹')
    xlim([1 img_size(2)]); ylim([1 img_size(1)])
    axis equal; box on;
    
    subplot(1,2,2); hold on;
    for i = 1:min(50, length(pred_tracks))
        track = pred_tracks{i};
        if ~isempty(track)
            plot(track(:,2), track(:,3), 'LineWidth', 1.2)
        end
    end
    title('预测轨迹') 
    xlim([1 img_size(2)]); ylim([1 img_size(1)])
    axis equal; box on;
end

function net = build_compatible_network(input_size)
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