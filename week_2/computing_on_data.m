A = [1 2; 3 4; 5 6];
B = [11 12; 13 14; 15 16];
C = [1 1; 2 2];

A*C

A.*B % element-wise multiplication, multiples each element of A by corresponding one in B
A.^2 % element-wise squaring

v = [1; 2; 3];
1./v % element-wise reciprocal
log(v) % element-wise log
exp(v) % element-wise exp()
abs([-1; 2; -3]) % element-wise absolute
-v

length(v)
v + ones(3, 1)

v' % transpose
max(v)
v < 2 % Returns v made of 1's and 0's

b = rand(3)
floor(b) % Rounds down to nearest int
ceil(b) % Rounds up to nearest int

D = magic(9);
sum(D, 1) % Sums up each column
sum(D, 2) % Sums up each row

% Sums up diagonal
D .* eye(9)
sum(sum(D .* eye(9))) % Sums up each column, then row