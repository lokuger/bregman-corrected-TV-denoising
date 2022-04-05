function d = kullback_leibler(f,g)
% computes pointwise kullback-leibler distance between f and g, i.e. the
% sum over g-f+f*log(f/g)
%
% requires f to be absolutely continuous wrt g, i.e. if g = 0, then f = 0
% further requires f and g to be of the same size


assert(all(size(f) == size(g)), 'input vectors/distributions must be of same size');
I = abs(f) > eps;
d = sum(g(I) - f(I) + f(I).*log(f(I)./g(I)),'all');
end
