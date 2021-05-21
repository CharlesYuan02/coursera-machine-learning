function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 
m = size(X, 1);

for i = 1:m
    features = zeros(p, 1);

    % For every element in the row vector, replace with X(i).^j
    for j = 1:p
        features(j) =  X(i).^j;
    end

    X_poly(i, :) = features;
% =========================================================================

end
