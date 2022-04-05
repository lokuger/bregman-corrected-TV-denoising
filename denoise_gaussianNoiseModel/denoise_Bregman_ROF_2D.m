function [U, iter] = denoise_Bregman_ROF_2D(g, mu, delta, varargin)
% function implements bregman iteration for L1-TV denoising of the 2D
% image g. Source of the algorithm:
% An iterative regularization method for total variation-based image
% restoration, Osher et al. 2005 Multiscale Model. Simul.
%
% Input:    g			-   noisy 2D image
%           mu			-   regularization parameter.
%           delta		-   list of noise levels for disc. principle stop crit
%							(if more than one, iterate until smallest and return output for every level)
%		varargin:
%			showIts		-	boolean whether iterates should be shown during iteration
%			verbose		-	boolean whether text output in console
% Output:   u			-   solution of the ROF model

%% check inputs, potentially assign discrepancy principle threshold
delta = sort(delta,'descend'); 
n_delta = length(delta);
delta_min = delta(end);
showIts = false; verbose = true;
for i = 1:2:length(varargin)
	switch varargin{i}
		case 'showIts', showIts = varargin{i+1};
		case 'verbose', verbose = varargin{i+1};
	end
end

[M,N] = size(g);
U = zeros(M,N,n_delta);


%% iteration
% ouputs
if showIts, figure(); end
if verbose
	fprintf('Start Bregman iterative denoising. Iterate until ||u-f||_2 < %.4f.\n',delta_min);
	fprintf('%5s\t|\t%12s\t|\t%12s\n','k','||u-f||_2','||v||_2'); fprintf([repmat('_',1,45),'\n']); 
end

u = zeros(M,N);
v = u;
iter = 0;
stopCrit = norm(u-g, 'fro') < delta_min;
if verbose, fprintf('%5d\t|\t%12.4g\t|\t%12.4g\n',iter,norm(u-g, 'fro'),norm(v, 'fro')); end
l = 0;	% l keeps track of the current discrepancy level. If a DP level is reached, l increases by 1
while ~ stopCrit
    iter = iter + 1;
    u = ROF_2D(g + v, 1/(2*mu));		% modified ROF. Regularization parameter is varied because Osher2005 and Chambolle2004 use different conventions
    v = v + g - u;						% update noise
	
	if showIts, imagesc(u); colormap(gray); pause(0.1); end

	resid = norm(u-g, 'fro');
    if verbose, fprintf('%5d\t|\t%12.4g\t|\t%12.4g\n',iter,resid,norm(v, 'fro')); end	% frobenius since this is the l2 norm if the images are viewed as vectors
	L = find(delta < resid, 1);
	if ~isempty(L) && L > l+1, U(:,:,l+1:L-1) = repmat(u,1,1,L-1-l); l = L-1; end	% saves iterates in U for all crossed DP levels
	stopCrit = isempty(L); % if L is empty, then the residuum is smaller than the smallest delta
end
U(:,:,l+1:end) = repmat(u,1,1,n_delta-l);
U = squeeze(U);
end