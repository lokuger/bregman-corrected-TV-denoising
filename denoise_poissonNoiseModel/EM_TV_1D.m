function u = EM_TV_1D(g, mu, omega, maxEMIts, tol)
% function implements EM-TV algorithm from 'EM-TV Methods for Inverse Problems
% with Poisson Noise', Sawatzky et al.; in 'Level Set and PDE-based
% Reconstruction Methods in Imaging', 2013 (ed. Burger & Osher)
%
% the idea of the algorithm is to solve the variational problem 
% min_u { D(Ku,g) + mu*TV(u) }
% where the data fidelity D is the Kullback-Leibler divergence and TV is
% total variation. Without TV, this is done by EM. The authors derive a
% forward-backward splitting resulting in alternating EM- and ROF-denoising
% steps.
% In a plain denoising problem, the EM step vanishes and we solve simply a
% sequence of weighted ROF problems
%
% Input:    g           -   noisy 1D signal
%           mu          -   regularization parameter
%           omega       -   damping of the TV step. omega in (0,1]. see
%                           paper for choice
%           maxEMIts    -   maximum number of iterations
%           tol         -   error tolerance for stopping criterion
%
% Output:   u           -   EM-TV solution

fprintf('EM-TV Poisson denoising algorithm started\n');
fprintf('%5s\t|\t%8s\t|\t%8s\t|\t%8s\n','k','opt_k','u_opt_k','p_opt_k');
fprintf([repmat('_',1,56),'\n'])

assert(size(g,1) == 1 || size(g,2) == 1, 'Input signal must be 1D');
g = reshape(g,[],1);
N = size(g,1);

u = mean(g)*ones(N,1); p = zeros(N,1);
k = 0;
stopcrit = false;
while ~stopcrit
    % if we solve a problem Ku = g, then this would be an EM step. 
    % But since this is denoising, K = Id, the first half step is just g
    ukhalf = g;
    ukhalf_damp = omega * ukhalf + (1-omega)*u;
    % compute the solution of the weighted, modified ROF. u is the denoised
    % signal, p is in the subgradient of TV at u
    [unew,pnew] = ROF_denoise_weighted_1D(ukhalf_damp, u, omega*mu);

    % print iterates and stopping criteria, check stopping criterion
    optk = weightedL2Norm(ones(N,1) - g./unew + mu*pnew, unew);
    uoptk = weightedL2Norm((unew - u)./(mu*u), unew);
    poptk = weightedL2Norm(mu*(pnew - p), unew);
    fprintf('%5d\t|\t%8.4g\t|\t%8.4g\t|\t%8.4g\n',k,optk,uoptk,poptk);
    stopcrit = (optk < tol && uoptk < tol && poptk < tol) || (k == maxEMIts);

    k = k+1;
    u = unew;
    p = pnew;
end
end

function r = weightedL2Norm(u,w)
% a weighted (squared) norm used only for the stopping criteria. See the paper, eq.
% (35) - (37)
% u is the vector whose norm is computed
% w is a weight vector
r = u' * (u.*w);
end