function p = computepdf_eps(x, rate, g, eps)
%COMPUTEPDF Compute the pdf of the eigenvalue distribution of bottom right
%subpopulation
%   Arguments:
%       x: points at which to evaluate the pdf.
%       rate: a vector of firing rates as the empirical rate distribution.
%       k: overall scaling factor.
%       g: variance parameter of the weight distribution.
%   Returns:
%       p: pdf values.

l = length(x);

G12 = zeros(2, l);
x0 = [-0.1*1i, -0.1*1i];

for i = 2:l
    G12(:,i) = fsolve(@(G) cauchyeq(x(i), rate, g, eps, G), x0,...
                  optimoptions('fsolve','Display','off'));
end
G = (G12(1,:)+G12(2,:))/2;

p = -imag(G)/pi;
p(p<0) = -p(p<0);

end

function f = cauchyeq(z, rate, g, eps, G)
%CAUCHYEQ The equation for the Cauchy transform G_c at z
G1 = G(1);
G2 = G(2);
sum = G1 + G2;
y = 2-z*sum;
x1 = 1-g^2/2*z*y;
x2 = 1-g^2/2*z*y - eps^2/2*z*(1-z*G2);

f1 = mean(1 ./(x1^2-z./rate))-(1-z*G1)/x1;
f2 = mean(1 ./(x2^2-z./rate))-(1-z*G2)/x2;

f = [f1, f2];
end

