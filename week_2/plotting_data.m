figure(1)
t = (0:0.01:0.98);
y1 = sin(2*pi * 4*t);
y2 = cos(2*pi * 4*t);
plot(t, y1);
hold on % Overlays figures
plot(t, y2);

xlabel('time');
ylabel('value');
legend('sin', 'cos');
title('My Plot');

% print -dpng 'myPlot.png'

% You can specify figure numbers
figure(2);
plot(t, y1);

figure(3);
plot(t, y2);

figure(4);
% Divides plot into a 1x2 grid, accesses first element
subplot(1, 2, 1); % Left side
plot(t, y1);
subplot(1, 2, 2); % Right side
plot(t, y2);