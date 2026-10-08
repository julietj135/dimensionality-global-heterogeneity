function p = computepdf(x, rate, g)
%COMPUTEPDF Compute the pdf of the eigenvalue distribution
%   Arguments:
%       x: points at which to evaluate the pdf.
%       rate: a vector of firing rates as the empirical rate distribution.
%       k: overall scaling factor.
%       g: variance parameter of the weight distribution.
%   Returns:
%       p: pdf values.

l = length(x);
p = zeros(1, l);

for i = 2:l
    p(i) = fsolve(@(G) cauchyeq(x(i), rate, g, G), -0.05*1i,...
                  optimoptions('fsolve','Display','off'));
end
p = -imag(p)/pi;
p(p<0) = -p(p<0);

end

function f = cauchyeq(z, rate, g, G)
%CAUCHYEQ The equation for the Cauchy transform G_c at z
y = 1-z*G;
x = 1 - g^2*z*y;
f = mean(1 ./ (x^2 - z./rate)) - y/x;

end

