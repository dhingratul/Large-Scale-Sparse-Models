%% RUn FOBOS Mini-batch version
% Input: X, Y, Lambda, Batch-size
% Output: W, obj
% Note: W, obj
function [ w, obj ] = fobos_minibatch( X, Y, lambda, batchsz )
% min_w (1/2n) || Y - Xw ||^2 + lambda || w ||_1

max_iter = 1e5;
a = 2;

[n, p] = size(X);

w = zeros(p, 1);
obj = zeros(1, max_iter);

for iter=1:max_iter
    
    rand_idx = randi(n, batchsz, 1);
    
    x = X(rand_idx, :);
    y = Y(rand_idx);
    
    grad = x' * (x * w - y) / batchsz;
    
    eta = a / sqrt(iter);
    
    w = w - eta * grad;
    
    w = wthresh(w, 's', eta * lambda);
    
    obj(iter) = 0.5 / n * (Y - X * w)' * (Y - X * w) + lambda * norm(w, 1);
    
    if mod(iter, 500) == 0
        fprintf('iter %d, obj = %g, sp = %d\n', iter, obj(iter), nnz(w));
    end
end

end

