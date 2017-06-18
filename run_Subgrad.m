clear;

load('gisette.mat', 'X_train', 'Y_train', 'X_test', 'Y_test');

lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 2.5, 3:10];

w = cell(1, length(lambdas));
acc = zeros(1, length(lambdas));
sp = zeros(1, length(lambdas));

for i=1:length(lambdas)
    fprintf('lambda = %g\n', lambdas(i));
    w{i} = subgrad(X_train, Y_train, lambdas(i));
    acc(i) = compute_acc(X_test, Y_test, w{i});
    sp(i) = nnz(w{i}) / length(w{i});
end

save('subgrad.mat', 'w', 'acc', 'sp');

%save('result/subgrad_minibatch.mat', 'w', 'obj', 'acc', 'sp');