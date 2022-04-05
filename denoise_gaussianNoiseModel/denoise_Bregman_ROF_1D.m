function [u, iter] = denoise_Bregman_ROF_1D(g, mu, delta)
% function implements bregman iteration for L1-TV denoising of the 1D
% signal g. Source of the algorithm:
% An iterative regularization method for total variation-based image
% restoration, Osher et al. 2005 Multiscale Model. Simul.
%
% Input:    g       -   noisy 1D signal
%           mu      -   regularization parameter
%           delta   -   noise level. needed for disc. principle stop crit
%                       If noise level is not set, use a variance estimate
% Output:   u       -   solution of the ROF model

%% check inputs, potentially assign discrepancy principle threshold
fprintf('Bregman iterative denoising started\n');

assert(size(g,1) == 1 || size(g,2) == 1, 'input signal must be 1D')
g = reshape(g,[],1);
N = size(g,1);

% if no noise level is set, use variance estimate
if nargin == 2
    delta = sqrt((N-1)*var(g)); 
    fprintf('no noise level provided, estimating discrepancy principle threshold to sqrt((N-1)*variance).\n');
    fprintf('Iterate until ||u-f||_2 < %.4f.\n',delta);
else
    fprintf('Iterate until ||u-f||_2 < %.4f.\n',delta);
end

%% iteration
u = zeros(N,1);
v = u;
k = 0;
fprintf('%5s\t|\t%12s\t|\t%12s\n','k','||u-f||_2','||v||_2');
fprintf([repmat('_',1,45),'\n'])
while norm(u - g) > delta
    k = k + 1;
    u = ROF_1D(g + v,mu);     % modified ROF
    v = v + g - u;                  % update noise

    fprintf('%5d\t|\t%12.4g\t|\t%12.4g\n',k,norm(u-g),norm(v));
end
end