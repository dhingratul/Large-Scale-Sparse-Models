%% Test Run- Lasso
clear;

% generate data
p = 20;
n = 100;
k = 2;

X = rand(n, p);
w = 10 * rand(p, 1);
w(randperm(p, p - k)) = 0;

Y = X * w + 0.01 * rand(n, 1);

% set lambda
lambda = 0.01;

% LARS solver, implemented by Matlab
w_lasso = lasso(X, Y, 'lambda', lambda);

% gradient descent solver
w_subgrad = subgrad(X, Y, lambda);

% compare non-zeros entries
figure;
hold on; grid on; box on;
plot(w_lasso, 'b', 'linewidth', 2);
plot(w_subgrad, 'r', 'linewidth', 2);

legend('LARS', 'Sub-Gradient');
