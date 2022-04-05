clear
close all

%% clean signal
N = 100;
f = ones(N,1);
f(48:52) = 2;

%% add Gaussian noise
sigma = 0.05;
noise = randn(N,1)*sigma;
f_noisy = f + noise;

%% solve using bregman iterated ROF
mu = 10;
delta = sqrt(N)*sigma;        % if noise has std sigma, this is the approximate L2 norm of the noise
u = denoise_Bregman_ROF_1D(f_noisy, mu,delta);

%% plots
figure()
hold on
plot(f,':k')
plot(f_noisy,':b')
plot(u, 'r');
axis([0 100 0 2.1])
hold off