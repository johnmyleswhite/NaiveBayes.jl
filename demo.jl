include("naive_bayes.jl")
include("bernoulli_nb.jl")

d = GaussianNaiveBayes([0.0 10.0;
	                    0.0 2.0],
	                   [1.0 1.0;
	                    1.0 1.0],
	                   [0.25, 0.75])

X, c = rand(d, 10_000)

d = fit(GaussianNaiveBayes, X, c)

predict(d, X)

loglikelihood(d, X, c)

logpdf(d, X[:, 1], c[1])
logpdf(d, X, c)

mean(d)

fit(GaussianNaiveBayes, rand(d, 1_000)...)

cor(X[:, c .== 1]')
cor(X[:, c .== 2]')

#

d = BernoulliNaiveBayes([0.1 0.9;
	                     0.9 0.1],
	                    [0.25, 0.75])

X, c = rand(d, 10_000)

d = fit(BernoulliNaiveBayes, X, c)

predict(d, X)

loglikelihood(d, X, c)

logpdf(d, X[:, 1], c[1])
logpdf(d, X, c)

mean(d)

fit(BernoulliNaiveBayes, rand(d, 1_000)...)

cor(X[:, c .== 1]')
cor(X[:, c .== 2]')
