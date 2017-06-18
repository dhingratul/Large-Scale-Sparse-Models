%% Compute the accuracy of the algorithm
% Input: X, Y, w
% Output: accuracy
% Note: Run from Run_ files
function [ acc ] = compute_acc( X, Y, w )
%check the sign of X*w and Y

acc = sum((X*w) .* Y > 0) / size(X, 1);

end

