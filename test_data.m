%% Test run file for subgradient-descent
clear;

load('gisette.mat', 'X_train', 'Y_train', 'X_test', 'Y_test');

tic;
% w = lasso(X_train, Y_train, 'lambda', 0.1);

w=subgrad(X_train,Y_train,0.1);
toc;
acc = compute_acc(X_test, Y_test, w);
fprintf('Accuracy by Lasso is %g, sparsity is %g\n', acc, nnz(w) / length(w));