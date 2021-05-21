function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.

% J = (1/m) * ((-y' * log(X * theta))) - (1 - y)' * log(1 - X * theta); 
% J = J + (lambda/(2*m)) * (sum(theta(2:end) .^ 2)); 

% Modified from logistic regression to linear regression
J = 1/(2*m) * ((X*theta) - y)' * ((X*theta) - y);
J = J + (lambda/(2*m)) * (theta(2:length(theta)))' * theta(2:length(theta));

% Same as ex3's
temp = theta;
temp(1) = 0;
grad = (1/m) * X' * ((X * theta) - y); 
grad = grad + (lambda / m) * temp;
grad = grad(:);
% =========================================================================

end
