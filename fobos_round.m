%% Run Fobos_round
% Input: X, Y, lambda, round
% Output: W, obj
% Note: Run from run_Fobos_round.m
function [ w, obj ] = fobos_round( X, Y, lambda, round )
% min_w (1/2n) || Y - Xw ||^2 + lambda || w ||_1

max_iter = 2e5;
a = 2;

[n, p] = size(X);

w = zeros(p, 1);
obj = zeros(1, max_iter);

for iter=1:max_iter
    
    rand_idx = randi(n);
    
    x = X(rand_idx, :);
    y = Y(rand_idx);
    
    grad = x' * (x * w - y);
    
    eta = a / sqrt(iter);
    
    w = w - eta * grad;
    
    if iter > round
        w = wthresh(w, 's', eta * lambda);
    end
    
    obj(iter) = 0.5 / n * (Y - X * w)' * (Y - X * w) + lambda * norm(w, 1);
    
    if mod(iter, 500) == 0
        fprintf('iter %d, obj = %g, sp = %d\n', iter, obj(iter), nnz(w));
    end
end

end

