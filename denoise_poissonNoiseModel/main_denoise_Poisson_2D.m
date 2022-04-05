clear
close all

%% clean signal
I_rgb = imread('~/Documents/lokuger.github.io/assets/images/fibo_box.jpg');
I = double(rgb2gray(I_rgb))';
I = imresize(I,0.2,'bicubic');	%downscale
[M,N] = size(I);

%% apply a poisson process
I_noisy = poissrnd(I);

%% solve using bregman iterated ROF
mu = 0.1;
% special case denoising: damping parameter can be stated explicitly
omega = 1.9*max(min(abs(I_noisy),[],'all'),1)^2/max(abs(I_noisy),[],'all')^2;
% omega = 1/2;
tol = 1e-5;
delta = kullback_leibler(I_noisy, I);
tau = 1.2;
maxBregIts = 3;
maxEMIts = 50;
u = bregman_EM_TV_denoise_2D(I_noisy,mu,omega,delta,1.5,maxBregIts,maxEMIts,tol);

%% plots
figure()
subplot(1,2,1)
imagesc(I'), pbaspect([M N 1]), colormap("gray"), axis off
% subplot(1,3,2)
% imagesc(I_noisy), pbaspect([3024 4032 1]), colormap("gray"), axis off
subplot(1,2,2)
imagesc(u'), pbaspect([M N 1]), colormap("gray"), axis off