% For, while loops
for i = 1:10
    v(i) = 2^i;
end
v

i = 1;
while i <= 5
    w(i) = 100;
    i = i + 1;
end
w

i = 1;
while true
    x(i) = 999;
    i = i + 1;
    if i == 6
        break
    end
end
x

% If else statements
t = 1;
if t == 1
    disp('The value is one');
elseif t == 2
    disp('The value is two');
else
    disp('The value is neither one nor two');
end

% Writing a function
squareThisNumber(2)

X = [1 1; 1 2; 1 3];
y = [1; 2; 3];
theta1 = [0; 1];
theta2 = [0; 0];
j1 = costFunctionJ(X, y, theta1)
j2 = costFunctionJ(X, y, theta2)

function y = squareThisNumber(x)
    y = x^2;
end

function J = costFunctionJ(X, y, theta)
    % X is the "design matrix" containing our training examples
    % y is the class labels
    
    m = size(X, 1); % Number of training examples
    predictions = X*theta; % Predictions of hypothesis on all m examples
    sqrErrors = (predictions-y) .^ 2; % Squared errors
    
    J = 1/(2*m) * sum(sqrErrors);
end