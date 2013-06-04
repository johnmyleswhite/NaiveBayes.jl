# X: Feature vector as a p x n matrix
# c: Vector of classes
# k: Number of distinct classes

# abstract BernoulliNaiveBayes
# abstract MultinomialNaiveBayes

# Implement alternative form for sparse matrices

# Use macros to make more generic, while getting high speed code

using Distributions

immutable GaussianNaiveBayes <: Distribution
	mu::Matrix{Float64} # Matrix of cluster centers: p x k
	sigma::Matrix{Float64} # Diagonals of covariance matrices for each class
	# Sigma is p x k: Sigma[d, c] is variance of dimension d for class c
	p::Vector{Float64} # Vector of probabilities for each class
	drawtable::Distributions.DiscreteDistributionTable
	function GaussianNaiveBayes(mu::Matrix, sigma::Matrix, p::Vector)
		new(mu, sigma, p, Distributions.DiscreteDistributionTable(p))
	end
end

function Distributions.rand(d::GaussianNaiveBayes)
	p = size(d.mu, 1)
	c = Distributions.draw(d.drawtable)
	x = Array(Float64, p)
	for dim in 1:p
		x[dim] = rand(Normal(d.mu[dim, c], d.sigma[dim, c]))
	end
	x, c
end

function Distributions.rand(d::GaussianNaiveBayes, n::Integer)
	p = size(d.mu, 1)
	X = Array(Float64, p, n)
	c = Array(Int, n)
	for obs in 1:n
		c[obs] = Distributions.draw(d.drawtable)
		for dim in 1:p
			X[dim, obs] = rand(Normal(d.mu[dim, c[obs]],
				                      d.sigma[dim, c[obs]]))
		end
	end
	X, c
end

function Distributions.mean(d::GaussianNaiveBayes)
	p, c = size(d.mu)
	mx = zeros(Float64, p)
	for cl in 1:c
		mx += d.p[cl] * d.mu[:, c]
	end
	mx /= c
	mc = 0.0
	for cl in 1:c
		mc += d.p[cl] * cl
	end
	mx, mc
end

function Distributions.fit(::Type{GaussianNaiveBayes}, X::Matrix, c::Vector)
	p, n = size(X)
	nclasses = max(c)
	mu = zeros(Float64, p, nclasses)
	sigma = zeros(Float64, p, nclasses)
	counts = zeros(Int, nclasses)
	for i in 1:n
		mu[:, c[i]] += X[:, i]
		counts[c[i]] += 1
	end
	for cl in 1:nclasses
		mu[:, cl] /= counts[cl]
	end
	for i in 1:n
		sigma[:, c[i]] += (X[:, i] - mu[:, c[i]]).^2
	end
	for cl in 1:nclasses
		sigma[:, cl] = sqrt(sigma[:, cl] / (counts[cl] - 1)) + 1e-8
	end
	return GaussianNaiveBayes(mu, sigma, counts / n)
end

function Distributions.logpdf(d::GaussianNaiveBayes, x::Vector, c::Real)
	p = length(x)
	res = log(d.p[c])
	for dim in 1:p
		res += logpdf(Normal(d.mu[dim, c], d.sigma[dim, c]), x[dim])
	end
	return res
end

function Distributions.logpdf(d::GaussianNaiveBayes, X::Matrix, c::Vector)
	p, n = size(X)
	res = zeros(Float64, n)
	for obs in 1:n
		res[obs] = log(d.p[c[obs]])
		for dim in 1:p
			res[obs] += logpdf(Normal(d.mu[dim, c[obs]],
				                      d.sigma[dim, c[obs]]),
			                   X[dim, obs])
		end
	end
	return res
end

function Distributions.loglikelihood(d::GaussianNaiveBayes, X::Matrix, c::Vector)
	p, n = size(X)
	res = 0.0
	for obs in 1:n
		res += logpdf(d, X[:, obs], c[obs])
	end
	return res
end

function predict(d::GaussianNaiveBayes, X::Matrix)
	nclasses = length(d.p)
	p, n = size(X)
	res = Array(Int, n)
	ll = Array(Float64, nclasses)
	for obs in 1:n
		for cl in 1:nclasses
			ll[cl] = logpdf(d, X[:, obs], cl)
		end
		res[obs] = indmax(ll)
	end
	return res
end
