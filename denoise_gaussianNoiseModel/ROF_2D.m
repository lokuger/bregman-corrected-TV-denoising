function [u,q] = ROF_2D(g,mu)
% function implements ROF model using primal-dual algorithm from 'An
% algorithm for total variation minimization and applications'. 
% Chambolle, A. Journal Math. Imag. and Vis. 2004
%
% solves:		argmin ||u-g||^2/(2*mu) + TV(u)
%
% the idea of the algorithm is to rewrite the ROF model problem in the dual
% formulation and reduce it to the computation of a nonlinear projection on
% the characteristic set of the cvx conjugate of TV.
%
% Input:    g   -   noisy image (passed as a M x N vector)
%           mu  -   regularization parameter
% Output:   u   -   solution of the ROF model
%           q   -   q is in the subgradient of TV at u
[M,N] = size(g);

px = zeros(M,N); py = px;
stopcrit = false;
tau = 1/8;
while ~stopcrit
    poldx = px; poldy = py;
    [rx,ry] = grad(div(px,py) - g/mu);
    px = (poldx + tau*rx)./(1 + tau*abs(rx));
    py = (poldy + tau*ry)./(1 + tau*abs(ry));
    stopcrit = max(max(abs(px-poldx),[],'all'),max(abs(py-poldy),[],'all')) < 1e-4;
end
q = div(px,py);
u = g - mu * q;

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