function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.

%{
Previously:
A = sigmoid(X * all_theta');
[~, p] = max(A, [], 2);
%}

% Now you have 2 layers corresponding to Theta1 and Theta2
X = [ones(m, 1) X]; % Add bias

L1 = sigmoid(X * Theta1');
L1 = [ones(m, 1) L1]; % Add bias

L2 = sigmoid(L1 * Theta2'); % L2 is output layer
[~, p] = max(L2, [], 2); 
% =========================================================================
end
