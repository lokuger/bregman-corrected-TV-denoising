function [u,q] = ROF_denoise_weighted_1D(g,h,mu)
% function implements weighted ROF model as in 'EM-TV Methods for Inverse 
% Problems with Poisson Noise', Sawatzky et al.; in 'Level Set and 
% PDE-based Reconstruction Methods in Imaging', 2013 (ed. Burger & Osher)
%
% This is the ROF model where the l2 norm is weighted with the components
% 1/h_i^2, see the chapter where it is analyzed. It can be solved as in
% Chambolle, 2005 by a projected-gradient algorithm in the dual space
%
% Input:    g   -   noisy 1D signal
%           h   -   vector of weights for the l2 norm
%           mu  -   regularization parameter
% Output:   u   -   solution of the ROF model
%           q   -   q is in the subgradient of TV at u

assert(size(g,1) == 1 || size(g,2) == 1, 'input signal must be 1D')
g = reshape(g,[],1); h = reshape(h,[],1);
assert(size(g,1) == size(h,1), 'input signal and weights must have the same size.')
N = size(g,1);

p = zeros(N,1);
stopcrit = false;
tau = 1/(8*max(abs(h)));
while ~stopcrit
    pold = p;
    r = grad(h.*div(p) - g/mu);
    p = (pold + tau*r)./(1 + tau*abs(r));
    stopcrit = norm(p-pold, "inf") < 1e-6;
end
q = div(p);
u = g - mu * h.* q;

end

function d = div(f)
% divergence of 1D signal as defined for this algorithm in Chambolle, 2005
d = zeros(size(f));
d(1) = f(1);
d(2:end-1) = f(2:end-1) - f(1:end-2);
d(end) = - f(end-1);
end

function g = grad(f)
% gradient of 1D signal as used in Chambolle, 2005. Neumann bdy cond
g = zeros(size(f));
g(1:end-1) = f(2:end) - f(1:end-1);
end