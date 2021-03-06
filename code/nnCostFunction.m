function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
% Part 3: Implement regularization with the cost function and gradients.
%

X = [ones(m, 1) X];						%size: 5000*401
a2 = sigmoid(Theta1*X');                %size: 25*5000

a2 = [ones(1, m); a2];                  %size: 26*5000
a3 = sigmoid(Theta2*a2);                %size: 10*5000

y01 = zeros(num_labels, m);
for j  = 1:m 							%binary y01
	y01(y(j), j) = 1;    
end

J = -1/m*(y01(:)'*log(a3(:)) + (1 - y01(:)')*log(1 - a3(:))) + lambda/(2*m)*...
	(Theta1(hidden_layer_size + 1:end)*Theta1(hidden_layer_size + 1:end)' + ...
	 Theta2(num_labels + 1:end)*Theta2(num_labels + 1:end)');

delta3 = a3 - y01;                      %size: 10*5000
delta2 = Theta2(:, 2:end)'*delta3.*a2(2:end, :).*(1-a2(2:end, :));    
											%size: 25*5000

Delta2 = delta3*a2';                    %size: 10*26
Delta1 = delta2*X;                      %size: 25*401

Theta2_grad = Delta2/m + lambda*[zeros(num_labels, 1) Theta2(:, 2:end)]/m;
Theta1_grad = Delta1/m + lambda*[zeros(hidden_layer_size, 1) Theta1(:, 2:end)]/m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
