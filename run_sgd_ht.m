clear;

load('gisette.mat', 'X_train', 'Y_train', 'X_test', 'Y_test');

Ks = [20 50 100 150 200 250 300 400 500];

w = cell(1, length(Ks));
obj = cell(1, length(Ks));
acc = zeros(1, length(Ks));
sp = zeros(1, length(Ks));

eta = 1;%2 / svds(X_train' * X_train, 1);
for i=1:length(Ks)
    fprintf('K = %g\n', Ks(i));
    [w{i}, obj{i}] = sgd_ht(X_train, Y_train, Ks(i), eta);
    acc(i) = compute_acc(X_test, Y_test, w{i});
    sp(i) = nnz(w{i}) / length(w{i});
end

save('result/sgd_ht.mat', 'w', 'obj', 'acc', 'sp', 'Ks');
