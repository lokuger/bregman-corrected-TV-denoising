function u = ROF_1D(g,mu)
% function implements ROF model using primal-dual algorithm from 'An
% algorithm for total variation minimization and applications'. 
% Chambolle, A. Journal Math. Imag. and Vis. 2004
%
% the idea of the algorithm is to rewrite the ROF model problem in the dual
% formulation and reduce it to the computation of a nonlinear projection on
% the characteristic set of the cvx conjugate of TV. The 2D version from
% Chambolle's paper is rewritten to fit the 1D case
%
% Input:    g   -   noisy 1D signal
%           mu  -   regularization parameter
% Output:   u   -   solution of the ROF model

assert(size(g,1) == 1 || size(g,2) == 1, 'input signal must be 1D')
g = reshape(g,[],1);
N = size(g,1);

p = zeros(N+1,1);
stopcrit = false;
tau = 1/4;
s = (g(2:end) - g(1:end-1))/mu;
while ~stopcrit
    pold = p(2:end-1);
    r = p(3:end) - 2*p(2:end-1) + p(1:end-2) - s;
    p(2:end-1) = (pold + tau*r)./(1 + tau*abs(r));
    stopcrit = norm(p(2:end-1)-pold, "inf") < 1e-6;
end
% denoised solution is u = g - mu*div(p)
u = g - mu * (p(2:end) - p(1:end-1));

end