%% Hard-Threshold
% Input: w, K
% Output: W
function [ w ] = HardThres_matlab( w, K )
% w: p x n matrix, n samples in p dimensions
% K: desired sparsity (for each column)

p = size(w);

if K >= p
    return;
end

[~, ind] = sort(abs(w), 'ascend');

w(ind(1:p-K)) = 0;

end

