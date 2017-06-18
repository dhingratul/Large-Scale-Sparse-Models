%% Test Run File- Lasso, Sub-gradient, mini-Subgradient
clear;

%addpath('algs');
load('gisette.mat')

% generate data
p = size(X_train,2);
n = size(X_train,1);
% p=20;
% n=100;
% k = 2;
%
% X_train = rand(n, p);
% w = 10 * rand(p, 1);
% w(randperm(p, p - k)) = 0;
%
% Y_train = X_train * w + 0.01 * rand(n, 1);

% set lambda
lambda = 0.1;

% LARS solver, implemented by Matlab
w_lasso = lasso(X_train, Y_train, 'lambda', lambda);

% gradient descent solver
w_subgrad = subgrad(X_train, Y_train, lambda);

batchSize=20;
w_miniSubgrad=miniSubgrad(X_train,Y_train,lambda,batchSize);

% compare non-zeros entries
figure;
hold on; grid on; box on;
plot(w_lasso, 'b', 'linewidth', 2);
plot(w_subgrad, 'r', 'linewidth', 2);
plot(w_miniSubgrad, 'g', 'linewidth', 2);

legend('LARS', 'Sub-Gradient', 'mini Sub-gradient');