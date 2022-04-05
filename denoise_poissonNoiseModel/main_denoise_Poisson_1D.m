clear
close all

%% clean signal
N = 100;
f = ones(N,1); f(20:26) = 2; f(40:60) = 1.8; f(75:90) = 0.6;
f = f*200;

%% apply a poisson process
f_noisy = poissrnd(f);

%% solve using bregman iterated ROF
mu = 1;
% special case denoising: damping parameter can be stated explicitly
omega = 1.9*norm(f_noisy,-Inf)^2/norm(f_noisy,Inf)^2;
% omega = 1/2;
tol = 1e-5;
delta = kullback_leibler(f_noisy, f);
tau = 1.2;
maxBregIts = 20;
maxEMIts = 200;
u = bregman_EM_TV_denoise_1D(f_noisy,mu,omega,delta,1.5,maxBregIts,maxEMIts,1e-6);

%% plots
figure()
hold on
plot(f,':k')
plot(f_noisy,':b')
plot(u, 'r');
axis([0 N 0 1.1*max(f_noisy)])
hold off