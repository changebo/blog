---
title: "Stochastic Bandits and UCB Algorithm"
date: 2018-12-08
description: An introduction to the stochastic bandit problem and the UCB Algorithm.
keywords: stochastic bandits, ucb, ucb1, upper confidence bound algorithm, ucb bandit
tags: ["algorithms", "machine learning"]
draft: false
---

In our recent paper *Vine Copula Structure Learning via Monte Carlo Tree Search* (AISTATS 2019), we apply the UCT (Upper Confidence bounds applied to Trees) algorithm to find an approximate solution to an NP-hard structure learning problem in statistics. 
The UCT algorithm is based on the UCB1 (Upper Confidence Bound) algorithm for the stochastic bandit problem.

Thankfully, I audited [Prof. Nick Harvey](https://www.cs.ubc.ca/~nickhar/)'s [learning theory course](https://www.cs.ubc.ca/~nickhar/F18-531/) this semester.
In one lecture, he gave a clear exposition of the multi-armed bandit problem and algorithms.
It deepened my understanding of the theoretical properties of the UCB1 algorithm.
This blog post is mostly a transcription of the lecture; it gives an overview of the regret bound analysis of the explore-first algorithm and UCB1 algorithm. The material is also based on the first chapter of Slivkins (2017).

## 1. Problem formulation

The [multi-armed bandit problem](https://en.wikipedia.org/wiki/Multi-armed_bandit) has many practical applications including clinical trials, financial portfolio design, and online advertising.
The name comes from imagining a gambler at a row of slot machines (sometimes known as "one-armed bandits"), each with a lever. 
In each round, the gambler picks a machine to play, then the reward for the chosen machine is observed.
The objective of the gambler is to maximize the sum of rewards earned through a sequence of lever pulls.
In the rest of the post, we use the term "arm" as a synonym for a slot machine.


In this post, we focus on the **stochastic bandit problem**, where the rewards for each arm are i.i.d. from a probability distribution specific to that arm.
Another variant of the multi-armed bandit problem is called the **adversarial bandit problem**, where an adversary chooses the reward for each arm while the player chooses an arm.


The fundamental tradeoff the gambler faces is between **exploration** and **exploitation**.
In machine learning, exploration stands for the acquisition of new knowledge, and exploitation refers to an optimized decision based on existing knowledge.
In the case of the multi-armed bandit problem, the gambler needs to decide between trying different arms to get more information about their rewards and playing the arm that has the highest average reward so far.



The following notations are used throughout the post.

* \\(K\\) is the total number of arms and \\(T\\) is the total number of rounds, both are known in advance. Arms are denoted by \\(a \in [K]\\), rounds by \\(t \in [T]\\). (The bracket notation \\([n]\\) denotes the first \\(n\\) positive integers \\([n] := \\{1, 2, \ldots, n\\}\\).)
* The reward for arm \\(a\\) is i.i.d. from \\(\mathcal{D}\_a\\), which is a distribution supported on \\([0,1]\\). The expected reward of arm \\(a\\) is denoted by \\(\mu(a) := \int\_0^1 x \, \mathrm{d}\mathcal{D}_a(x)\\).
* The best expected reward is denoted by \\(\mu^* := \max\_{a \in [K]} \mu(a)\\), and the best arm is \\(a^* = \operatorname{argmax}\_{a \in [K]} \mu(a)\\).

The (cumulative) **regret** in round \\(t\\) is defined as
$$ R(t) =  \mu^* t  - \sum\_{s=1}^t \mu(a_s), $$
where \\(a_s\\) is the chosen arm in round \\(s\\).
The regret is the difference between the sum of rewards associated with the optimal arm and the sum of expected rewards by an algorithm.
The goal of the algorithm is to minimize regret.
Note that \\(a_s\\) is a random quantity, since it may depend on the randomness in rewards and/or the algorithm.
Therefore the regret \\(R(t)\\) is also random and we are interested in the expected regret \\(\mathbb{E}[R(t)]\\) or \\(\mathbb{E}[R(T)]\\).


It may be a good time now to review [Hoeffding's inequality](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality), an important concentration inequality widely used in machine learning theory.

> **Hoeffding's inequality**  
> Let \\(X_1, \ldots, X_n\\) be independent random variables and let \\(\bar{X}=\frac{1}{n}\sum\_{i=1}^n X\_i\\). Assume \\(0 \leq X_i \leq 1\\) almost surely. Then, for any \\(t>0\\),
> $$\mathbb{P}(|\bar{X} - \mathbb{E}[\bar{X}]| > t) \leq 2 \exp (-2nt^2).$$


## 2. Explore-first algorithm

A simple idea is to explore arms uniformly and pick an empirically best arm for exploitation.

> **Explore-first algorithm**  
> 
> 1. Exploration phase:     Try each arm \\(N\\) times. Let \\(\bar\mu(a)\\) be the average reward for arm \\(a\\);  
> 2. Exploitation phase: Select arm \\(\hat{a}\\) with the highest average reward \\(\hat{a} = \operatorname{argmax}_{a \in [K]} \bar\mu(a)\\). Play arm \\(\hat{a}\\) in all remaining rounds.

The parameter \\(N\\) is fixed in advance; it will be chosen later as a function of \\(T\\) and \\(K\\).



### 2.1 Regret bound

Our goal is to give an upper bound of the expected regret \\(\mathbb{E}[R(T)]\\) as a function of \\(K\\) and \\(T\\):
$$
\mathbb{E}[R(T)]=O \big(T^{2 / 3} (K \log T)^{1 / 3} \big).
$$

To facilitate our analysis, we define the **clean event** \\(C\\) to be the event that \\(\bar\mu(a)\\) is close to \\(\mu(a)\\) for all arms:
$$
C = \\{ |\bar\mu(a) - \mu(a)| \leq r, \forall a \in [K] \\},
$$
where \\(r\\) is called the **confidence radius**.
The **bad event** \\(\bar{C}\\) is the complement of the clean event.
By the [law of total expectation](https://en.wikipedia.org/wiki/Law_of_total_expectation),
$$
\mathbb{E}[R(T)] = 
\mathbb{E}[R(T) | C]
\mathbb{P}[C]
+
\mathbb{E}[R(T) | \bar{C}]
\mathbb{P}[\bar{C}]. 
$$

First of all, we want to make sure \\(\mathbb{P}[\bar{C}]\\) is small.
In other words, the average reward \\(\bar\mu(a)\\) should be a good estimate of the true expected reward \\(\mu(a)\\). 
By Hoeffding's inequality, the deviation of the average reward from the true expected reward can be quantified as follows:

$$
\mathbb{P}( |\bar\mu(a) - \mu(a)| > r) \leq 2 \exp (-2Nr^2), \quad \forall a \in [K].
$$

We want to bound this probability by a sufficiently small number, say \\(2/T^4\\). This can be achieved by setting the confidence radius \\(r = \sqrt{\frac{2 \log T}{N}}\\):
$$
\mathbb{P}( |\bar\mu(a) - \mu(a)| > r) \leq \frac{2}{T^4}, \quad \forall a \in [K].
$$
Note that the choice of \\(T^4\\) is somewhat arbitrary; the exponent only affects the multiplicative constant of \\(r\\). 


Using a [union bound](https://en.wikipedia.org/wiki/Boole%27s_inequality), we can find an upper bound of the probability of the bad event.
$$
\begin{align}
\mathbb{P}(\bar{C})
&=
\mathbb{P} \big(\\{ \exists a \in [K] \ \text{s.t.}\ |\bar\mu(a) - \mu(a)| > r \\} \big)
\newline
&=
\mathbb{P} \big(\bigcup_{a \in [K]}\\{|\bar\mu(a) - \mu(a)| > r \\} \big)
\newline
&\leq \sum\_{a \in [K]} \mathbb{P} (|\bar\mu(a) - \mu(a)| > r )
\newline
&\leq
\frac{2K}{T^4}
\newline
&\leq
\frac{2}{T^3}.
\end{align}
$$
The last inequality is because each arm is explored at least once, i.e., \\(T \geq K\\).


Next, we focus on \\(\mathbb{E}[R(T) | C]\\).
By definition, \\(\bar\mu(\hat{a}) \geq \bar\mu(a^*)\\).
Given the clean event happens, 
an upper bound of \\(\mu(a^\*) - \mu(\hat{a}) \\) can be obtained:
$$
\begin{align}
\mu(a^\*) - \mu(\hat{a}) 
&\leq 
\mu(a^\*) - \mu(\hat{a}) + \bar\mu(\hat{a}) - \bar\mu(a^\*)
\newline
&= 
\big(\mu(a^\*) - \bar\mu(a^\*)\big) + \big(\bar\mu(\hat{a}) - \mu(\hat{a}) \big)
\newline
&\leq 2r.
\end{align}
$$
Intuitively, it indicates that conditioning on the clean event, the chosen arm \\(\hat{a}\\) cannot be much worse than \\(a^\*\\).


There are \\(NK\\) rounds in the exploration phase, and each round has at most regret of 1; there are \\(T - NK\\) rounds in the exploitation phase and the regret incurred in each round is bounded by \\(2r\\).
As a result, the regret in round \\(T\\) can be bounded by
$$
\begin{align}
\mathbb{E}[R(T) | C] 
&\leq NK + (T - NK) 2r
\newline
&\leq NK + 2Tr
\newline
&= NK + 2T \sqrt{\frac{2 \log T}{N}}.
\end{align}
$$
Since we can choose the number of rounds in the exploration phase,
the right-hand side can be minimized by setting 
\\( N = (T/K)^{2 / 3} (2\log T)^{1 / 3} \\).
Therefore, 
$$
\mathbb{E}[R(T) | C] 
\leq
T^{2 / 3} (2K\log T)^{1 / 3}. 
$$

So far, we have 

* \\(\mathbb{E}[R(T) | C] 
\leq
T^{2 / 3} (2K \log T)^{1 / 3}
\\);
* \\( \mathbb{P}[C] \leq 1 \\);
* \\( \mathbb{E}[R(T) | \bar{C}] \leq T \\), since there are in total \\(T\\) rounds, each round incurs at most regret of 1;
* \\(\mathbb{P}[\bar{C}] \leq 2 / T^4 \\).

Putting everything together,
$$
\begin{align}
\mathbb{E}[R(T)] &= 
\mathbb{E}[R(T) | C]
\mathbb{P}[C]
+
\mathbb{E}[R(T) | \bar{C}]
\mathbb{P}[\bar{C}] 
\newline
&\leq
T^{2 / 3} (2K\log T)^{1 / 3}
+
T \cdot \frac{2}{T^3}
\newline
&= O \big(T^{2 / 3} (K \log T)^{1 / 3} \big).
\end{align}
$$



## 3. Upper confidence bound algorithm

The problem with the explore-first algorithm is that each arm is explored for the same number of rounds, which causes inefficiency.
In other words, the exploration schedule should depend on the history of the observed rewards. 
Instead of using the same confidence radius for any arm in any round, we denote the confidence radius for arm \\(a\\) at time \\(t\\) by \\(r_t(a)\\).
Let \\(n_t(a)\\) be the number of times arm \\(a\\) is selected in rounds \\(1, 2, \ldots, t\\) and \\(\bar\mu\_t(a)\\)  be the average reward of arm \\(a\\) up to time \\(t\\). 
The **upper confidence bound** is defined as
$$
\mathrm{UCB}\_t(a) = \bar\mu\_t(a) + r\_t(a),
$$
where \\( r_t(a) = \sqrt{\frac{2 \log T}{n_t(a)}} \\).


The UCB1 algorithm chooses the best arm based on an optimistic estimate. The algorithm is as follows. 

> **UCB1 algorithm**
> 
> 1. Try each arm once;
> 1. In round \\(t\\), pick \\( a_t = \operatorname{argmax}\_{a \in [K]} \mathrm{UCB}\_t(a)\\).  

Unlike the explore-first algorithm, there is no clear exploration/exploitation phase. However, the definition of the upper confidence bound manifests the exploration-exploitation tradeoff: \\(\bar\mu\_t(a)\\) encourages the exploitation of high reward arms, while \\(r\_t(a)\\) encourages the exploration of less played arms.


### 3.1 Regret bound

The regret bound of the UCB1 algorithm is
$$
\mathbb{E}[R(t)] = O \big(\sqrt{Kt \log T} \big), 
\quad
\forall t \in [T].
$$
The idea of the analysis is the same as before: define a clean event \\(C\\), obtain upper bounds of \\(\mathbb{P}[\bar{C}]\\) and \\(\mathbb{E}[R(t)|C]\\), and finally bound \\(\mathbb{E}[R(t)]\\).

Since the confidence radius now depends on \\(t\\) as well, we need a more refined definition of the **clean event**:
$$
C = \\{|\bar\mu_t(a) - \mu(a)| \leq r_t(a), \forall a \in [K], \forall t \in [T] \\}.
$$
Applying Hoeffding's inequality and a union bound (and ignoring some technicalities), 
$$
\mathbb{P}[\bar{C}] \leq \frac{2KT}{T^4} \leq \frac{2}{T^2}.
$$



Now we focus on \\(\mathbb{E}[R(t)|C]\\) and assume the clean event holds.
By definition, \\( \mathrm{UCB}\_t(a_t) \geq \mathrm{UCB}\_t(a^*) \\) for any round \\(t \in [T]\\). 
As a result, 

$$
\begin{align}
\mu(a^\*) - \mu(a\_t) 
&\leq 
\mu(a^\*) - \mu(a\_t) + \mathrm{UCB}\_t(a_t) - \mathrm{UCB}\_t(a^*)
\newline
&= \big(\mu(a^\*) - \mathrm{UCB}\_t(a^\*)\big) + \big(\mathrm{UCB}\_t(a\_t) - \mu(a\_t) \big),
\end{align}
$$
where 
$$
\mu(a^\*) - \mathrm{UCB}\_t(a^\*)
=\mu(a^\*) - \bar\mu\_t(a^\*) - r\_t(a^\*)
\leq 0,
$$
and 
$$
\mathrm{UCB}\_t(a\_t) - \mu(a\_t)
=\bar\mu\_t(a\_t) - \mu(a\_t) + r\_t(a\_t)
\leq 2 r\_t(a\_t).
$$
Therefore,
$$
\mu(a^\*) - \mu(a\_t) 
\leq
2 r\_t(a\_t)
= 2 \sqrt{\frac{2 \log T}{n_t(a_t)}}.
$$

For each arm \\(a\\) and a given time \\(t \in [T]\\), consider the last round \\(t_0 \leq t\\) when this arm is chosen.
Since arm \\(a\\) is never played between round \\(t_0\\) and round \\(t\\), we have \\(n\_{t\_0}(a) = n_t(a)\\).
Applying the above inequality to arm \\(a\\) and round \\(t_0\\), 
$$
\mu(a^\*) - \mu(a) \leq 2 \sqrt{\frac{2 \log T}{n_t(a)}},
\quad
\forall t \in [T].
$$
Intuitively it means that, under the clean event, if an arm is selected many times, it cannot be much worse than the best arm; or equivalently, if an arm is much worse than the best arm, it won't be selected many times.


The regret in round \\(t\\) is thus bounded by
$$
R(t) = \sum\_{a \in [K]}
n_t(a) \big(\mu(a^\*) - \mu(a)\big)
\leq 
2 \sqrt{2 \log T}
\sum\_{a \in [K]}
\sqrt{n_t(a)}.
$$
Since the square root function is concave, by [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality),
$$
\sum\_{a \in [K]}
\sqrt{n_t(a)}
= K \sum\_{a \in [K]}
\frac{1}{K} \sqrt{n_t(a)}
\leq 
K 
\sqrt{\frac{1}{K}\sum\_{a \in [K]} n_t(a)}
= \sqrt{Kt}.
$$
Therefore,
$$
\mathbb{E}[R(t)|C] \leq 2 \sqrt{2 Kt \log T}.
$$

Finally, much like what we did for the explore-first algorithm, the expected regret in round \\(t\\) can be bounded by
$$
\begin{align}
\mathbb{E}[R(t)] &= 
\mathbb{E}[R(t) | C]
\mathbb{P}[C]
+
\mathbb{E}[R(t) | \bar{C}]
\mathbb{P}[\bar{C}] 
\newline
&\leq
2 \sqrt{2 Kt \log T}
+
t \cdot \frac{2}{T^2}
\newline
&= O \big(\sqrt{Kt \log T} \big), 
\quad
\forall t \in [T].
\end{align}
$$


### 3.2 An instance-dependent regret bound

We can also obtain another regret bound using the inequality 
$$
\mu(a^\*) - \mu(a) \leq 2 \sqrt{\frac{2 \log T}{n_T(a)}}.
$$
Rearrange the terms,
$$
n_T(a) \leq \frac{8 \log T}{(\mu(a^\*) - \mu(a))^2},
\quad
\text{if }
\mu(a) < \mu(a^\*).
$$
The regret in round \\(T\\) is bounded by
$$
R(T) = \sum\_{a \in [K]} n\_T(a) \big(\mu(a^\*) - \mu(a)\big)
\leq 
8 \log T
\sum\_{\\{a \in [K]: \mu(a) < \mu(a^\*)\\}}
\frac{1}{\mu(a^\*) - \mu(a)}.
$$
Therefore,
$$
\mathbb{E}[R(T)]
\leq 
O(\log T)
\sum\_{\\{a \in [K]: \mu(a) < \mu(a^\*)\\}}
\frac{1}{\mu(a^\*) - \mu(a)}.
$$

This regret bound is logarithmic in \\(T\\), with a constant that can be arbitrarily large depending on a problem instance.
This type of regret bound is called **instance-dependent**. The existence of a logarithmic regret bound is a benefit of the UCB1 algorithm compared to the explore-first algorithm, whose regret bound is polynomial in \\(T\\).


## References

- Slivkins, A. (2017). Introduction to Multi-Armed Bandits. http://slivkins.com/work/MAB-book.pdf.
