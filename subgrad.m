%% Sub-gradient Descent
% Input: X, Y, Lambda
% Output: W
% Note: Run from run_Subgrad.m
function [ w ] = subgrad( X, Y, lambda )
% min_w (1/2n) || Y - Xw ||^2 + lambda || w ||_1

max_iter = 1e5;
eta = 2;

[n, p] = size(X);

%idx = randi(n, max_iter);

w = zeros(p, 1);

for iter=1:max_iter
    %     rand_idx = idx(iter);
    %     x = X(rand_idx, :);
    %     y = Y(rand_idx);
    
    grad = X' * (X * w - Y) / n + lambda * sign(w);
    
    w = w - eta / iter * grad;
end

end

