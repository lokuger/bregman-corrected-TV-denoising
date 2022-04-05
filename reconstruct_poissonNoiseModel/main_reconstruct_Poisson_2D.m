clear
close all
addpath(genpath('../../flexbox'))

%% clean signal
N = 32;
angles = 1:3:180;
I = 1000*phantom('Modified Shepp-Logan',N); I(I<0) = 0;		% there might be negative (machine-precision magnitude) values in phantom I

%% create Radon matrix and generate noisy signal
% [R,size_g] = generateRadonMatrix(size(I), angles);
R = eye(numel(I)); size_g = size(I);
g = reshape(R*I(:),size_g);
g_noisy = poissrnd(g);

%% solve using bregman iterated ROF
mu = 0.5;
omega = 0.5; % damping parameter choice is pretty much try-and-error
tol = 1e-3;																	
delta = kullback_leibler(g_noisy(:), R*I(:));
tau = 2;
maxBregIts = 10;
maxEMIts = 1000;
u = bregman_EM_TV_2D(g_noisy, R, N, N, mu, omega, delta, tau, maxBregIts, maxEMIts, tol);

%% plots
figure()
subplot(1,3,1)
imagesc(I), pbaspect([128 128 1]), colormap("gray"), axis off
subplot(1,3,2)
imagesc(g_noisy), pbaspect([128 128 1]), colormap("gray"), axis off
subplot(1,3,3)
imagesc(u), pbaspect([128 128 1]), colormap("gray"), axis off