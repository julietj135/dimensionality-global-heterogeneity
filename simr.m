function [R, mu, sigma] = simr(targetMean, targetPR, N)
%SIMR Sample firing rate distribution R
[mu, sigma] = lognparam(targetMean, targetPR);
R = lognrnd(mu, sigma, N, 1);

end