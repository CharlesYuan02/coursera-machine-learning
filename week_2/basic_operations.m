% To display command window, Home -> Environment -> Layout -> 
% Command Window Only

5+6
3-2
5*8
1/2
2^6
1 == 2
1 ~= 2 % Does not equal
1 && 0 % And
1 || 0 % Or

a = 3;
b = "hi";
c = (3 >= 1);
a, b, c

d = 3.1416;
disp(sprintf('2 decimals: %0.2f', d)) 
fprintf('2 decimals: %0.2f\n', d) % Same thing

A = [1 2; 3 4; 5 6];
A

v = [1 2 3];
transpose(v)

v2 = (1:0.1:2);
v2

v3 = 2*ones(2, 3);
v3

v4 = zeros(1, 3);
v4

w = rand(3, 3); % Rand between 0 and 1
w

w2 = -6 + sqrt(10)*(randn(1, 10000));
% histogram(w)

size(A)

% pwd Shows current directory
% cs 'directory' to change directory
% load 'filename'
% save hello.txt v -ascii (saves vector as txt file)

A
A(:, 2) % Second column
A(2, :) % Second row
A = [A, [100; 101; 102]]; % Appends another column vector to right
A

A(:) % Put all elements of A into a single column vector

B = [11 12; 13 14; 15 16];
C = [17 18; 19 20; 21 22];
D = [B C] % Concatenate two matrices beside each other
E = [B; C] % Stack two matrices on top of each other
