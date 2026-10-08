function [p1,p2] = computepdf_eps_Gs(x, rate, g, eps)
%COMPUTEPDF Compute the pdf of the eigenvalue distribution
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

p1 = -imag(G12(1,:))/pi;
p1(p1<0) = -p1(p1<0);

p2 = -imag(G12(2,:))/pi;
p2(p2<0) = -p2(p2<0);

end

function f = cauchyeq(z, rate, g, eps, G)
%CAUCHYEQ The equation for the Cauchy transform G_c at z
G1 = G(1);
G2 = G(2);
sum = G1 + G2;
y = 2-z*sum;
x1 = 1-g^2/2*z*y;
f1 = mean(1 ./(x1^2-z./rate))-(1-z*G1)/x1;

x2 = 1-g^2/2*z*y - eps^2/2*z*(1-z*G2);
f2 = mean(1 ./(x2^2-z./rate))-(1-z*G2)/x2;

f = [f1, f2];

% G1_y = 2-z*(G(3)+G(4));
% G1_yy = 2-z*(G(1)+G(2));
% G1_x1 = 1-z*g^2/2*G1_yy;
% G1_x2 = 1-z*g^2/2*G1_y;
% f1 = mean(1 ./(G1_x1*G1_x2-z./rate))-(1-z*G(1))/G1_x1;
% 
% G2_x1 = 1-z*g^2/2*G1_yy-z*eps^2/2*(1-z*G(2));
% G2_x2 = 1-z*g^2/2*G1_y-z*eps^2/2*(1-z*G(4));
% f2 = mean(1 ./(G2_x1*G2_x2-z./rate))-(1-z*G(2))/G2_x1;
% 
% G3_x1 = 1-z*g^2/2*G1_y;
% G3_x2 = 1-z*g^2/2*G1_yy;
% f3 = mean(1 ./(G3_x1*G3_x2-z./rate))-(1-z*G(3))/G3_x1;
% 
% G4_x1 = 1-z*g^2/2*G1_y-z*eps^2/2*(1-z*G(4));
% G4_x2 = 1-z*g^2/2*G1_yy-z*eps^2/2*(1-z*G(2));
% f4 = mean(1 ./(G4_x1*G4_x2-z./rate))-(1-z*G(4))/G4_x1;
% 
% f = [f1, f2, f3, f4];

end

