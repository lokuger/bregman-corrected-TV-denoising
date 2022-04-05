clear
close all

%% clean signal
N = 128;
sl = phantom('Modified Shepp-Logan', N);

%% add Gaussian noise
sigma = 0.02;
noise = randn(N)*sigma;
sl_noisy = sl + noise;

%% solve using bregman iterated ROF
mu = 0.5;
delta = N*sigma;        % if noise has std sigma, this is the approximate Frobenius norm of the noise (L2 norm if image is seen as vector)
u = denoise_Bregman_ROF_2D(sl_noisy, mu,delta);

%% plots
figure()
subplot(1,3,1)
imagesc(sl), colormap(gray)
subplot(1,3,2)
imagesc(sl_noisy), colormap(gray)
subplot(1,3,3)
imagesc(u), colormap(gray)