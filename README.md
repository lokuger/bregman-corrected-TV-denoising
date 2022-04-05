# Bregman corrected TV denoising 
TV denoising algorithms using Bregman bias correction techniques for different noise models

## Denoising for Gaussian noise
This case is solved using Bregman iterations as proposed in [1] by repeatedly solving modified versions of the ROF model with a corrected noise term.
In a subroutine, the ROF model [2] is solved using Chambolle's algorithm [3] that reduces the problem to the computation of a nonlinear projection in the dual space. Code is provided for the 1D and the 2D problem.

## Image reconstruction in the presence of Poisson noise
In the case of Poisson noise (and in particular for low counts), the data fidelity term must be the Kullback-Leibler divergence, hence the ROF model is not a good basis for a denoising or reconstruction algorithm. Instead, the signal/image is denoised using the Bregman FB-EM-TV algorithm. Reconstruction is carried out by a forward-backward scheme, splitting the iteration into an EM step and a TV correction step with a weight function in the TV functional. Bias correction is carried out by employing a few Bregman iterations.

## Denoising for Poisson noise
The denoising case is a special case of the image reconstruction case above, only differing from the latter by the vanishing EM step.


## References
1: Osher, S.; Burger, M.; Goldfarb, D.; Xu, J.; Yin, W. An Iterative Regularization Method for Total Variation-Based Image Restoration 
Multiscale Modeling; Simulation, Society for Industrial and Applied Mathematics, 2005, 4, 460-489 

2: Rudin, L. I.; Osher, S.; Fatemi, E. Nonlinear total variation based noise removal algorithms. 
Physica D: Nonlinear Phenomena, 1992, 60, 259-268 

3: Chambolle, A. An Algorithm for Total Variation Minimization and Applications
Journal of Mathematical Imaging and Vision, 2004, 20, 89-97 

4: Sawatzky, A.; Brune, C.; Kösters, T.; Wübbeling, F. &amp; Burger, M. EM-TV Methods for Inverse Problems with Poisson Noise
in: Burger, M.; Mennucci, A. C.; Osher, S.; Rumpf, M. (Eds.). Level Set and PDE Based Reconstruction Methods in Imaging, 
Springer International Publishing, 2013, 71-142 
