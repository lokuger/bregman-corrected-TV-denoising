function [u,q] = ROF_denoise_weighted_2D(g,h,mu)
% function implements weighted ROF model as in 'EM-TV Methods for Inverse 
% Problems with Poisson Noise', Sawatzky et al.; in 'Level Set and 
% PDE-based Reconstruction Methods in Imaging', 2013 (ed. Burger & Osher)
%
% This is the ROF model where the l2 norm is weighted with the components
% 1/h_i^2, see the chapter where it is analyzed. It can be solved as in
% Chambolle, 2005 by a projected-gradient algorithm in the dual space
%
% Input:    g   -   noisy image (passed as a M x N vector)
%           h   -   vector of weights for the l2 norm (M x N)
%           mu  -   regularization parameter
% Output:   u   -   solution of the ROF model
%           q   -   q is in the subgradient of TV at u

assert(all(size(g) == size(h)), 'input signal and weights must have the same size.')
[M,N] = size(g);

px = zeros(M,N); py = px;
stopcrit = false;
tau = 1/(8*max(abs(h),[],'all'));
while ~stopcrit
    poldx = px; poldy = py;
    [rx,ry] = grad(h.*div(px,py) - g/mu);
    px = (poldx + tau*rx)./(1 + tau*abs(rx));
    py = (poldy + tau*ry)./(1 + tau*abs(ry));
    stopcrit = max(max(abs(px-poldx),[],'all'),max(abs(py-poldy),[],'all')) < 1e-4;
end
q = div(px,py);
u = g - mu * h.* q;

end

function d = div(fx,fy)
% divergence of 2D signal as defined for this algorithm in Chambolle, 2005
assert(all(size(fx)==size(fy)),'invalid input to divergence operator')
d = zeros(size(fx));
d(1,:) = fx(1,:);
d(2:end-1,:) = fx(2:end-1,:) - fx(1:end-2,:);
d(end,:) = - fx(end-1,:);
d(:,1) = d(:,1) + fy(:,1);
d(:,2:end-1) = d(:,2:end-1) + (fy(:,2:end-1) - fy(:,1:end-2));
d(:,end) = d(:,end) - fy(:,end-1);
end

function [fx,fy] = grad(f)
% gradient of 2D signal as used in Chambolle, 2005. Neumann bdy cond
fx = zeros(size(f));
fy = fx;
fx(1:end-1,:) = f(2:end,:)-f(1:end-1,:);
fy(:,1:end-1) = f(:,2:end)-f(:,1:end-1);
end