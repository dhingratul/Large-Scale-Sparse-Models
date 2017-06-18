%% Mini-batch Sub-gradient Descent
% Input: X, Y, Lambda, batch-Size
% Output: W
% Note: Run from run_Subgrad_minibatch.m

function [ w ] = miniSubgrad( X, Y, lambda, batchSize)
% min_w (1/2n) || Y - Xw ||^2 + lambda || w ||_1

max_iter = 1e5;
eta = 2;

[n, p] = size(X);

%idx = randi(n, max_iter);

w = zeros(p, 1);

for iter=1:max_iter
    rand_idx=randi(n,[batchSize,1]);
    %     rand_idx = idx(iter);
    %     x = X(rand_idx, :);
    %     y = Y(rand_idx);
    
    grad = X(rand_idx,:)' * (X(rand_idx,:) * w - Y(rand_idx)) / batchSize + lambda * sign(w);
    
    w = w - eta / iter * grad;
end

end
