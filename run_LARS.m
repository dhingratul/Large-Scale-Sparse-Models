clear;

addpath('spams/src_release/');
addpath('spams/build');


load('gisette.mat', 'X_train', 'Y_train', 'X_test', 'Y_test');

% LARS
lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 2.5, 3:10, 20:10:100, 150:50:1000, 1100:100:5000];
param.mode = 0;e

for i=1:length(lambdas)
    param.lambda = lambdas(i);
    
    w{i} = mexLasso(Y_train, X_train, param);
    acc(i) = compute_acc(X_test, Y_test, w{i});
    sp(i) = nnz(w{i}) / length(w{i});
end

save('result/LARS.mat', 'w', 'acc', 'sp');