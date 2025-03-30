%% 1. Generate Data
numSteps = 1000;
tx = linspace(0, 10*pi, numSteps);
data = sin(tx) + 0.1*randn(size(tx));

% Visualize data
figure
plot(data)
title('Training Data - Noisy Sine Wave')
xlabel('Time Step')
ylabel('Amplitude')

%% 2. Data Preprocessing
lookback = 20;  % Input sequence length for training
[XTrain, YTrain] = createSequences(data, lookback);

% Standardize the data
dataMean = mean(data);
dataStd = std(data);
XTrain = (XTrain - dataMean)/dataStd;
YTrain = (YTrain - dataMean)/dataStd;

% Convert training data into the proper LSTM format:
% [1 x lookback x 1 x N] → [N x 1 cell] where each cell is 1x20
XTrain = squeeze(num2cell(XTrain, [1 2]));
XTrain = cellfun(@(x) reshape(x, [1, lookback]), XTrain, 'UniformOutput', false);
YTrain = squeeze(YTrain);

%% 3. Define Network Architecture
layers = [ ...
    sequenceInputLayer(1, 'Name', 'input')  % Feature dimension = 1; time steps determined by data
    lstmLayer(128, 'Name', 'lstm1', 'OutputMode', 'last') 
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(1, 'Name', 'output') 
    regressionLayer('Name', 'regression')];

%% 4. Set Training Options with Gradient Clipping
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'GradientThreshold', 1, ...  % Prevent gradient explosion
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

%% 5. Train the Network
net = trainNetwork(XTrain, YTrain, layers, options);

%% 6. Multi-step Prediction Demo
testStart = 800;  % Start index for prediction
testInput = data(testStart:testStart+lookback-1);
numPredictions = 200;

% Standardize the input
testInputNorm = (testInput - dataMean)/dataStd;

% Initialize the sliding window (maintains the last 20 normalized points)
currentInput = reshape(testInputNorm, [1, lookback]);
predictions = zeros(numPredictions, 1);

for i = 1:numPredictions
    % Predict using the current window (wrapped as a cell)
    pred = predict(net, {currentInput});
    
    % Convert prediction back to original scale
    predictions(i) = pred * dataStd + dataMean;
    
    % Update the input window: remove the oldest value and append the new normalized prediction
    currentInput = [currentInput(:, 2:end), pred];
end

%% 7. Visualization of Predictions with Prediction Interval
figure
hold on
plot(data, 'b-', 'LineWidth', 1.5)
plot(testStart+lookback:testStart+lookback+numPredictions-1, predictions, 'r-', 'LineWidth', 1.5)
fill([testStart+lookback, testStart+lookback+numPredictions, testStart+lookback+numPredictions, testStart+lookback],...
     [min(data), min(data), max(data), max(data)], 'y', 'FaceAlpha', 0.1, 'EdgeColor', 'none')
legend('Original Data', 'Predicted Series', 'Prediction Interval')
title('LSTM Multi-step Forecast')
xlabel('Time Step')
ylabel('Amplitude')
grid on

%% Helper Function to Create Sequences
function [X, Y] = createSequences(data, lookback)
    numSamples = length(data) - lookback;
    X = zeros(1, lookback, 1, numSamples);  % Dimensions: [channels, sequenceLength, batchSize, numSamples]
    Y = zeros(1, 1, numSamples);
    
    for i = 1:numSamples
        X(1, :, 1, i) = data(i:i+lookback-1);
        Y(1, 1, i) = data(i+lookback);
    end
end
