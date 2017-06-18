%% Stochastic Gradient Descent with Hard Thresholding
% Input: X, Y, k, eta
% Output: w, obj
% Note: Run from Run_ files
function [ w, obj ] = sgd_ht( X, Y, K, eta )
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
    
    %eta = a / sqrt(iter);
    
    w = w - eta * grad;
    
    w = mex_HardThres(w, K); % use w = HardThres_matlab(w, K) if the mex_HardThres does not work
    
    obj(iter) = 0.5 / n * (Y - X * w)' * (Y - X * w);
    
    if mod(iter, 500) == 0
        fprintf('iter %d, obj = %g\n', iter, obj(iter));
    end
end

end