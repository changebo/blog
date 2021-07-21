---
title: "Importance Weighted Autoencoders and Jackknife Methods"
date: 2019-08-09
description: Importance Weighted Autoencoders and Jackknife Methods
keywords: Variational Inference, Variational Autoencoders, VAE, Evidence Lower Bound, ELBO, Importance Weighted Autoencoders, IWAE, Jackknife
tags: ["machine learning", "statistics"]
draft: false
---

This blog post is a summary of two papers: Burda et al. (2015) and Nowozin (2018).
We first give a quick overview of variational inference and variational autoencoders (VAE), which approximate the posterior distribution by a simpler distribution and maximize the evidence lower bound (ELBO).
Blei et al. (2017) and Zhang et al. (2018) are among some excellent survey papers on variational inference.
Next, the importance weighted autoencoder (IWAE) is introduced, and its properties are presented.
Finally, we describe the jackknife variational inference (JVI) as a way to reduce the bias of IWAE estimators.

## Variational autoencoders


Consider a Bayesian model involving the (set of) latent variable $z$ and  observation $x$, where the joint density can be decomposed into
$$
p(x, z) = p(x | z) p(z),
$$
where $p(x | z)$ is the likelihood and $p(z)$ is the prior distribution of the latent variable $z$.
The posterior distribution $p(z|x)$ is of central interest in Bayesian inference. 
However, it is often intractable, and approximate inference is required.


Variational inference aims to approximate the posterior distribution by a variational distribution and to derive a lower bound of the marginal log-likelihood of data $\log p(x)$.
Variational autoencoder (VAE) is a type of amortized variational inference method.
Here, ''amortized'' means that the variational distribution $q(z|x)$ is parametrized by a function of $x$, whose parameters are shared across all observations.


We first rewrite the marginal log-likelihood by introducing the variational distribution $q(z|x)$:
\begin{align}
\log p(x) &= \log \int p(x|z) p(z) \mathrm{d} z
\newline
&= \log \int \frac{p(x|z) p(z)}{q(z|x)} q(z|x) \mathrm{d} z
\newline
&= \log \mathbb{E}\_{q(z|x)} \left[ \frac{p(x|z) p(z)}{q(z|x)} \right].
\label{eq:marginal}
\tag{1}
\end{align}
If we pull the expectation out of the logarithm function, which is concave, [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality) gives the following inequality:
$$
\log p(x) \ge 
\mathbb{E}\_{q(z|x)} 
\log \left[ \frac{p(x|z) p(z)}{q(z|x)} \right].
\label{eq:elbo}
\tag{2}
$$
The right-hand side is called the evidence lower bound (ELBO), denoted by $\mathcal{L}$.
The inference problem then becomes an optimization problem that tries to find a variational distribution $q(z|x)$ that maximizes $\mathcal{L}$.


There are at least three ways of rewriting the ELBO:
\begin{align}
\mathcal{L} &=
\log p(x) - D\_{\mathrm{KL}} ( q(z|x) \parallel p(z|x) )
\newline
&=
\mathbb{E}\_{q(z|x)} \log p(x,z) + \mathbb{E}\_{q(z|x)} [-\log q(z|x)]
\newline
&=
\mathbb{E}\_{q(z|x)} \log p(x|z) - D\_{\mathrm{KL}} ( q(z|x) \parallel p(z) )
\end{align}

* The first equation shows that the difference between the marginal log-likelihood $\log p(x)$ and the ELBO $\mathcal{L}$ is the KL divergence between $q(z|x)$ and $p(z|x)$. 
When these two distributions are identical (almost everywhere), the marginal log-likelihood equals the ELBO.

* In the second equation, the first term represents the "energy" and the second term represents the entropy of $q(z|x)$. The energy term encourages $q(z|x)$ to focus probability mass on where the joint probability $p(x, z)$ is large. The entropy encourages $q(z|x)$ to spread the probability mass to avoid concentrating on one location.

* The third equation is a more explicit representation of the standard architecture of a variational autoencoder.
The variational distribution $q(z|x)$ and the likelihood function $p(x|z)$ are represented by an encoder network and a decoder network, respectively. 
Furthermore, $q(z|x)$ and $p(z)$ are often assumed to be multivariate independent Gaussian so that their KL divergence is of closed-form.


A simple Monte Carlo estimator of the ELBO $\mathcal{L}$ approximates the expectation in Equation \ref{eq:elbo} by the sample mean. 
Let $z_i$, for $i=1, \ldots, k$, be independent samples drawn from $q(z|x)$, then the estimator is
$$
\widehat{\mathcal{L}}\_k^{\mathrm{ELBO}} :=
\frac{1}{k} \sum\_{i=1}^k \log \left[ \frac{p(x|z\_i) p(z\_i)}{q(z\_i|x)} \right].
$$
It is obvious that the estimator is unbiased, i.e., $\mathbb{E}\_{z\_i \sim q(z|x)} \widehat{\mathcal{L}}\_k^{\mathrm{ELBO}} = \mathcal{L}$.


## Importance weighted autoencoders

What we have described so far is first to define the ELBO $\mathcal{L}$ as a lower bound of $\log p(x)$, and then to estimate it by $\widehat{\mathcal{L}}\_k^{\mathrm{ELBO}}$.
An alternative approach is to approximate the expectation (inside the logarithm function) in Equation \ref{eq:marginal} by Monte Carlo, which leads to the importance weighted autoencoders (IWAE) estimator:
$$
\widehat{\mathcal{L}}\_k^{\mathrm{IWAE}} :=
\log \left[ \frac{1}{k} \sum\_{i=1}^k  \frac{p(x|z\_i) p(z\_i)}{q(z\_i|x)} \right].
$$
Note the difference between $\widehat{\mathcal{L}}\_k^{\mathrm{ELBO}}$ and $\widehat{\mathcal{L}}\_k^{\mathrm{IWAE}}$.


If we denote $\mathcal{L}\_k := \mathbb{E}\_{z\_i \sim q(z|x)} \widehat{\mathcal{L}}\_k^{\mathrm{IWAE}}$, then by Jensen's inequality,
$$
\mathcal{L}\_k 
\le \log \left[ \mathbb{E}\_{z\_i \sim q(z|x)} \frac{1}{k} \sum\_{i=1}^k  \frac{p(x|z\_i) p(z\_i)}{q(z\_i|x)} \right]
= \log p(x).
$$
In other words, the expectation of $\widehat{\mathcal{L}}\_k^{\mathrm{IWAE}}$ is also a lower bound of $\log p(x)$.
When $k=1$, the ELBO and IWAE estimators are equivalent. 
It can be shown that $\mathcal{L}\_k$ is tighter than $\mathcal{L}$ when $k>1$:
$$
\mathcal{L} = \mathcal{L}\_1 \le \mathcal{L}\_2 \le \cdots \le \log p(x),
$$
and
$$
\lim_{k \to \infty} \mathcal{L}\_k = \log p(x).
$$
Unsurprisingly, $\widehat{\mathcal{L}}\_k^{\mathrm{IWAE}}$ also converges in probability to $\log p(x)$ as $k\to \infty$.
A more detailed asymptotic analysis shows that
$$
\mathcal{L}\_k 
= \log p(x) - \frac{\mu\_2}{2 \mu^2} \frac{1}{k} + \left( \frac{\mu\_3}{3\mu^2} - \frac{3\mu\_2^2}{4\mu^4} \right) \frac{1}{k^2} + O(k^{-3}),
$$
where $\mu$ and $\mu\_j$ are the expectation and the $j$-th central moment of $p(x|z\_i) p(z\_i) / q(z\_i|x)$ with $z\_i \sim q(z|x)$, respectively.

An interesting perspective on the IWAE is that, $\widehat{\mathcal{L}}\_k^{\mathrm{IWAE}}$ can be regarded as an estimator of $\log p(x)$.
As shown above, the estimator is [consistent](https://en.wikipedia.org/wiki/Consistent_estimator) but [biased](https://en.wikipedia.org/wiki/Bias_of_an_estimator), and the bias is in the order of $O(k^{-1})$.
The remaining sections try to reduce the bias to a higher/smaller order so that the estimator is closer to the marginal log-likelihood when $k$ is large.


## Jackknife resampling

The [jackknife](https://en.wikipedia.org/wiki/Jackknife_resampling) is a resampling technique that can be used to estimate the bias of an estimator and further to reduce the bias. 
Let $\widehat{T}\_n$ be a consistent but biased estimator of $T$, evaluated on $n$ samples.
Assume the expectation $\mathbb{E} (\widehat{T}\_n)$ can be written as an asymptotic expansion as $n \to \infty$:
$$
\mathbb{E} (\widehat{T}\_n) = T + \frac{a_1}{n} + \frac{a_2}{n^2} + O(n^{-3}).
$$
Then the bias of $\widehat{T}\_n$ is in the order of $O(n^{-1})$.


A debiased estimator $\widetilde{T}\_{n,1}$ can be defined as follows:
$$
\widetilde{T}\_{n,1} := n \widehat{T}\_n - (n-1) \widehat{T}\_{n-1}.
$$
The idea is that the first order term is canceled by calculating the difference.
\begin{align}
\mathbb{E} (\widetilde{T}\_{n,1})
&= n \mathbb{E} (\widehat{T}\_n) - (n-1) \mathbb{E} (\widehat{T}\_{n-1})
\newline
&= n \left( T + \frac{a_1}{n} + \frac{a_2}{n^2} + O(n^{-3}) \right)
\newline
&\qquad - (n-1) \left( T + \frac{a_1}{n-1} + \frac{a_2}{(n-1)^2} + O(n^{-3}) \right)
\newline
&= T + \frac{a_2}{n} - \frac{a_2}{n-1} + O(n^{-2})
\newline
&= T + O(n^{-2}).
\end{align}
The bias of $\widetilde{T}\_{n,1}$ is in the order of $O(n^{-2})$ instead of $O(n^{-1})$. 
When $n$ is large, $\widetilde{T}\_{n,1}$ has a lower bias than $\widehat{T}\_n$.

The estimator $\widehat{T}\_{n-1}$ can be calculated on any $n-1$ samples. In practice, given $n$ samples, it is evaluated on the $n$ "leave-one-out" subsets of size $n-1$, and the average of the $n$ estimates is used in place of $\widehat{T}\_{n-1}$, which reduces the variance of the estimator.

The above debiasing method can be further generalized to higher orders. 
For example, let
$$
\widetilde{T}\_{n,2} 
:= \frac{n^2}{2} \widehat{T}\_n - (n-1)^2 \widehat{T}\_{n-1} + \frac{(n-2)^2}{2} \widehat{T}\_{n-2},
$$
then
$$
\mathbb{E} ( \widetilde{T}\_{n,2} )
= T + O(n^{-3}),
$$
that is, the bias of $\widetilde{T}\_{n,2}$ is in the order of $O(n^{-3})$.
More generally, for
$$
\widetilde{T}\_{n,m}
:= \sum\_{j=0}^m c(n, m, j) \widehat{T}\_{n-j},
$$
where
$$
c(n, m, j) = (-1)^j \frac{(n-j)^m}{(m-j)! j!},
$$
the bias is in the order of $O(n^{-(m+1)})$.


## Jackknife variational inference

The application of the jackknife method to the IWAE estimator should be straightforward.
The jackknife variational inference (JVI) estimator is defined as follows:
$$
\widehat{\mathcal{L}}\_{k,1}^{\mathrm{JVI}} 
:= k \widehat{\mathcal{L}}\_k^{\mathrm{IWAE}} - (k-1) \widehat{\mathcal{L}}\_{k-1}^{\mathrm{IWAE}},
$$
and more generally,
$$
\widehat{\mathcal{L}}\_{k,m}^{\mathrm{JVI}} 
:= \sum\_{j=0}^m c(k, m, j) \widehat{\mathcal{L}}\_{k-j}^{\mathrm{IWAE}}.
$$
The bias of $\widehat{\mathcal{L}}\_{k,m}^{\mathrm{JVI}}$, as an estimator of $\log p(x)$, is thus in the order of $O(k^{-(m+1)})$.

Again, the IWAE estimator $\widehat{\mathcal{L}}\_{k-j}^{\mathrm{IWAE}}$ can be evaluated on a single subset of samples of size $k-j$, or by the average of that on all subsets of size $k-j$.
In the latter case, the computational cost is significant since $\sum_{j=0}^m {k \choose j}$ could be large; the time complexity is bounded by
$$
O \left( k e^m \left( \frac{k}{m} \right)^m \right).
$$
In practice, the algorithm is feasible only for small values of $m$. 
Other variations of JVI are also provided by Nowozin (2018), at the cost of higher variance of the estimator.


## References

* Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. Journal of the American Statistical Association, 112(518), 859-877.

* Burda, Y., Grosse, R., & Salakhutdinov, R. (2015). Importance weighted autoencoders. International Conference on Learning Representations.

* Nowozin, S. (2018). Debiasing evidence approximations: On importance-weighted autoencoders and jackknife variational inference. International Conference on Learning Representations.

* Zhang, C., Butepage, J., Kjellstrom, H., & Mandt, S. (2018). Advances in variational inference. IEEE transactions on pattern analysis and machine intelligence.
