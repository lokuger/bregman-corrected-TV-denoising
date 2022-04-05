function u = bregman_EM_TV_denoise_1D(g, mu, omega, delta, tau, maxBregIts, maxEMIts, tol)
% function implements bregman-EM-TV scheme for denoising 1D signals
% corrupted by Poisson noise. Algorithm as in 'Bregman-EM-TV Methods with
% Application to Optical Nanoscopy', Sawatzky et al. 2009, Lecture Notes in
% Computer Science
%
% Idea of the algorithm is to bregmanize the EM-TV scheme. This leads to
% an alternating EM-TV denoising step and a weighted, modified ROF model.
%
% Input:    g           -   noisy 1D signal
%           mu          -   regularization parameter
%           omega       -   damping parameter
%           delta       -   noise level. needed for disc. principle stop crit
%           tau         -   scaling of delta disc princ
%           maxBregIts  -   maximum number outer bregman iterations
%           maxEMIts    -   maximum number inner EM iterations
%           tol         -   error tolerance in EM iteration scheme
% Output:   u           -   solution of the ROF model

fprintf('Starting Bregman-EM-TV algorithm!\n')
fprintf('Running until KL(f,u) < delta*tau = %4g\n',delta*tau);

assert(size(g,1) == 1 || size(g,2) == 1, 'Input signal must be 1D');
g = reshape(g,[],1);
N = size(g,1);

l = 0;
u = mean(g)*ones(N,1);
v = zeros(N,1);
stopBregConv = kullback_leibler(g, u) < tau*delta;
stopBregIter = false;
fprintf('%1s\t|\t%8s\t|\t%8s\n','k','KL(f, u_l)','||v_l||_2');
while ~stopBregConv && ~stopBregIter
    % inner EM iteration
    k = 0;
    stopEMiter = false;
    stopEMconv = false; optk = Inf; uoptk = Inf; poptk = Inf; p = zeros(N,1);
    while ~stopEMiter && ~ stopEMconv
        % for details of this inner loop see EM_TV_1D.m
        ukhalf = g;
        ukhalf_damp = omega * (ukhalf + v.*u) + (1-omega)*u;
        [unew,pnew] = ROF_denoise_weighted_1D(ukhalf_damp, u, omega*mu);

        k = k+1;
        optk = weightedL2Norm(ones(N,1) - g./unew + mu*pnew, unew);
        uoptk = weightedL2Norm((unew - u)./(mu*u), unew);
        poptk = weightedL2Norm(mu*(pnew - p), unew);
%         fprintf('%5d\t|\t%8.4g\t|\t%8.4g\t|\t%8.4g\n',k,optk,uoptk,poptk);
        stopEMconv = optk < tol && uoptk < tol && poptk < tol;
        stopEMiter = k == maxEMIts;
    
        u = unew;
        p = pnew;
    end
    if stopEMconv
        fprintf(['\tInner EM-iteration converged after %d iterations at\n' ...
            '\topt_k = %.4g\tu_opt_k = %.4g\tp_opt_k = %.4g\n'],k,optk,uoptk,poptk);
    else
        fprintf(['\tInner EM iteration failed, aborted after %d iterations at\n' ...
            '\topt_k = %.4g\tu_opt_k = %.4g\tp_opt_k = %.4g\n'],k,optk,uoptk,poptk);
    end

    % Bregman: update noise/dual component by which we shift in the mod ROF
    v = v - (ones(N,1) - g./u);

    % update for next iteration and print in command line
    l = l+1;
    fprintf('%1d\t|\t%8.4g\t|\t%8.4g\n',l,kullback_leibler(g, u),norm(v,2));
    stopBregConv = kullback_leibler(g, u) < tau*delta;
    stopBregIter = l == maxBregIts;
end
end

function r = weightedL2Norm(u,w)
% a weighted (squared) norm used only for the stopping criteria. See the paper, eq.
% (35) - (37)
% u is the vector whose norm is computed
% w is a weight vector
r = u' * (u.*w);
end