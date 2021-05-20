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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
X = [ones(m,1) X]; % Bias
z2 = Theta1 * X'; 
a2 = sigmoid(z2);

a2 = [ones(m,1) a2']; % Add bias
z3 = Theta2 * a2'; 

% Recode labels as vectors containing only 0 or 1
temp = zeros(num_labels, m);
for i = 1:m
    temp(y(i), i)=1;
end

% Unvectorized
J = (1/m) * sum(sum((-temp) .* log(sigmoid(z3)) - (1 - temp) .* log(1 - sigmoid(z3))));

% Avoid regularizing bias terms
t1 = Theta1(:, 2:size(Theta1, 2));
t2 = Theta2(:, 2:size(Theta2, 2));

% Regularization
J = J + lambda * (sum(sum(t1 .^ 2)) + sum(sum(t2 .^ 2))) / (2*m);

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
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% Refer to pages 4-5 of Cost Function and Backpropagation pdf
for j = 1:m
    a1 = (X(j, :))';
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    
    a2 = [1; a2]; % Add bias (26x1 now)
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    delta_3 = a3 - temp(:, j);
    z2 = [1; z2]; % Add bias (26x1)
    
    delta_2 = (Theta2' * delta_3) .* (sigmoid(z2) .* (1 - sigmoid(z2)));
    
    delta_2 = delta_2(2:end); % Remember, we don't find delta_1!
    Theta2_grad = Theta2_grad + delta_3 * a2';
    Theta1_grad = Theta1_grad + delta_2 * a1';
end

Theta2_grad = (1/m) * Theta2_grad;
Theta1_grad = (1/m) * Theta1_grad;
 
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

% Page 5
% If j = 0:
% Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m;
% Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m;

% If j != 0:
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end));
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end));

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
% =========================================================================
end
