% Initialization
clear ; close all; clc

% Setup the parameters
input_layer_size  = 560;            % Input Images of Digits
hidden_layer_size = 25;             % 25 hidden units
num_labels = 5;                     % 10 labels, from 1 to 5   
                                    % (note that we have mapped "0" to label 5)

% Load Data
fprintf('Loading Data ...\n')
load('face.mat');

% Normalize X
[X_norm, mu, sigma] = featureNormalize(X);

% Show image as seen by the classifier
imshow(X_norm, [-1, 1] );

% Randomly select 1179 data points to train
sel = randperm(size(X_norm, 1));
sel_train = sel(1:1179);                                  % 1179 = 1965*0.6
X_train = X_norm(sel_train, :);                           % choose training set
y_train = y(sel_train);

X_cv = X_norm(sel(1180:1572), :);
y_cv = y(sel(1180:1572));

X_test = X_norm(sel(1573:end), :);
y_test = y(sel(1573:end));

% Initializing Pameters
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Implement Backpropagation
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

% Training NN
fprintf('\nPress enter to continue, Training Neural Network... \n')
pause;

options = optimset('MaxIter', 1000);
accuracy = zeros(2, 1);
theta1 = zeros(input_layer_size, hidden_layer_size);
theta2 = zeros(hidden_layer_size, num_labels);
Lambda = 0;

for lambda = 0.01:0.01:1.8

	% Create "short hand" for the cost function to be minimized
	costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

	% CostFunction is a function that takes in only one argument (the
	% neural network parameters)
	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

	% Obtain Theta1 and Theta2 back from nn_params
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

	% Implement Predict
	pred = predict(Theta1, Theta2, X_cv);
	fprintf('\nCross validation Set Accuracy: %f\n', mean(double(pred == y_cv))*100);
	accuracy(2) = mean(double(pred == y_cv))*100;

	if accuracy(2) >= accuracy(1)
		theta1 = Theta1;
		theta2 = Theta2;
		Lambda = lambda;
		accuracy(1) = accuracy(2);
	end
end

% Get the Accuracy in Test set
pred = predict(theta1, theta2, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == y_test))*100);
