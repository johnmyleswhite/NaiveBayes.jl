immutable BernoulliNaiveBayes <: Distribution
	mu::Matrix{Float64} # Matrix of cluster centers: p x k
	p::Vector{Float64} # Vector of probabilities for each class
	drawtable::Distributions.DiscreteDistributionTable
	function BernoulliNaiveBayes(mu::Matrix, p::Vector)
		new(mu, p, Distributions.DiscreteDistributionTable(p))
	end
end

function Distributions.rand(d::BernoulliNaiveBayes)
	p = size(d.mu, 1)
	c = Distributions.draw(d.drawtable)
	x = Array(Int, p)
	for dim in 1:p
		x[dim] = rand(Bernoulli(d.mu[dim, c]))
	end
	x, c
end

function Distributions.rand(d::BernoulliNaiveBayes, n::Integer)
	p = size(d.mu, 1)
	X = Array(Int, p, n)
	c = Array(Int, n)
	for obs in 1:n
		c[obs] = Distributions.draw(d.drawtable)
		for dim in 1:p
			X[dim, obs] = rand(Bernoulli(d.mu[dim, c[obs]]))
		end
	end
	X, c
end

function Distributions.mean(d::BernoulliNaiveBayes)
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

function Distributions.fit(::Type{BernoulliNaiveBayes}, X::Matrix, c::Vector)
	p, n = size(X)
	nclasses = max(c)
	mu = zeros(Float64, p, nclasses)
	counts = zeros(Int, nclasses)
	for i in 1:n
		mu[:, c[i]] += X[:, i]
		counts[c[i]] += 1
	end
	for cl in 1:nclasses
		mu[:, cl] /= counts[cl]
	end
	return BernoulliNaiveBayes(mu, counts / n)
end

function Distributions.logpdf(d::BernoulliNaiveBayes, x::Vector, c::Real)
	p = length(x)
	res = log(d.p[c])
	for dim in 1:p
		res += logpdf(Bernoulli(d.mu[dim, c]), x[dim])
	end
	return res
end

function Distributions.logpdf(d::BernoulliNaiveBayes, X::Matrix, c::Vector)
	p, n = size(X)
	res = zeros(Float64, n)
	for obs in 1:n
		res[obs] = log(d.p[c[obs]])
		for dim in 1:p
			res[obs] += logpdf(Bernoulli(d.mu[dim, c[obs]]), X[dim, obs])
		end
	end
	return res
end

function Distributions.loglikelihood(d::BernoulliNaiveBayes,
	                                 X::Matrix,
	                                 c::Vector)
	p, n = size(X)
	res = 0.0
	for obs in 1:n
		res += logpdf(d, X[:, obs], c[obs])
	end
	return res
end

function predict(d::BernoulliNaiveBayes, X::Matrix)
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
