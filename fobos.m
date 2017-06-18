%% FOBOS Algorithm
% Input: X, Y, Lambda
% Output: W, Obj
% Note: Run from run_Fobos.m
function [ w, obj ] = fobos( X, Y, lambda )
% min_w (1/2n) || Y - Xw ||^2 + lambda || w ||_1

max_iter = 2e5;
a = 2;

[n, p] = size(X);

w = zeros(p, 1);
obj = zeros(1, max_iter);

for iter=1:max_iter
    %     if mod(iter, 200) == 0
    %         fprintf('iter %d, obj = %g\n', iter, obj(iter));
    %     end
    
    rand_idx = randi(n);
    
    x = X(rand_idx, :);
    y = Y(rand_idx);
    
    grad = x' * (x * w - y);
    
    eta = a / sqrt(iter);
    
    w = w - eta * grad;
    
    w = wthresh(w, 's', eta * lambda);
    
    obj(iter) = 0.5 / n * (Y - X * w)' * (Y - X * w) + lambda * norm(w, 1);
    
    if mod(iter, 500) == 0
        fprintf('iter %d, obj = %g\n', iter, obj(iter));
    end
end

end

