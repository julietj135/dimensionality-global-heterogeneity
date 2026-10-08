function [R, E, temp] = simrc(W, targetMean, targetPR)
%SIMRC Sample a pair of firng rate R and covariance matrix C
N = size(W, 1);
R = simr(targetMean, targetPR, N);
temp = diag(R.^(-1/2)) - W;
E = svd(temp).^(-2);

end