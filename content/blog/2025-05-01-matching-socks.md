+++
date = "2025-05-01T00:36:15-04:00"
title = "Matching Socks"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = []
+++

Are you the type of person to optimize their morning routine? I ask because some of my friends have optimized their morning routine to a tee. In fact, they only purchase the same pair of socks to stock their drawers. This way they never need to fumble around to find matching pairs of socks. Personally, I don't mind doing this that much. What gets me is the fact that I don't need to spend nearly as much time as I do folding laundry. Amortized across days finding matching socks is fine but on laundry day, you get hit with the full-on cost of having diversity in your sock collection. Thinking about this made me remember something from some of my classes at Harvard...

We do have a way to analyze the average number of socks you need to look at before you find a matching pair though. I first encountered this in lecture during my time at Harvard in STAT210 taught by Prof. Joe Blitzstein. The formal problem statement is as follows (don't worry it's short):

**Problem**: *You are getting ready for school and in your drawer there are $2n$ individual socks (specifically $n$ distinct pairs of socks). How many draws from this drawer do you need to find your first matching pair of socks on average?*

It's quite a challenging problem if you're just thinking of it for the first time. The solution that I was taught is as follows:

**Proof**: Let's biject the $2n$ socks, denoted as $S$, as uniform and IID random variables on the interval $[0, 1]$ (think of it as the time we draw the $i$-th sock). Let's denote the socks corresponding to pair $i$ as $S_i$ and $S_{i+1}$. The time at which we completely draw the pair $i$ is equivalently $T_i = \max (S_i, S_{i+1})$. Following this notation, the time at which we completely draw _any_ pair is $C = \min(T_1, T_2, \dots, T_n)$. We are interested in finding the expectation of this quantity. We will attempt to relate the expectation in count space to the expectation in time space shortly.  

As an aside, remember that if $U_1, U_2 \sim \text{Unif}(0, 1)$, then $\max(U_1, U_2) \sim \text{Beta}(2, 1)$. We also know that generally for $U \sim \text{Unif}(0, 1)$:
$$U^{\frac{1}{\alpha}} \sim \text{Beta}(\alpha, 1)$$

This implies that $C = \min(U_1^\frac{1}{2}, \dots, U_n^\frac{1}{2}) = \min(U_1, \dots, U_n)^\frac{1}{2}$. Again, referring to our order statistics trivia facts:
$$\min(U_1, \dots, U_n) \sim \text{Beta}(1, n)$$
This implies that $C \sim \text{Beta}(1, n)^\frac{1}{2}$, with some abuse of notation hehe:

$$E[C] = \frac{1}{\beta(1, n)} \int_0^1 x^\frac{1}{2} (1 - x)^{n-1} dx = \frac{\beta(\frac{3}{2}, n)}{\beta(1, n)}$$

Alternatively, we view $C$ as a sum of disjoint intervals. Denote $T$ as the number of socks we need to draw until a pair is drawn. Then, (trivially) $C = S_{T}$. Continuing the order statistics route, where $S_{(i)}$ denotes the $i$-th order statistic:
$$C = \sum_{i=1}^T S_{(i + 1)} - S_{(i)}$$

In this setting, $S_{(i + 1)} - S_{(i)} \sim \text{Beta}(1, n)$. Since $T$ is a stopping time, we can apply Wald's lemma. This means that $E[C] = E[T] \cdot E[S_{(i+1)} - S_{(i)}] = E[T] \cdot \frac{1}{n + 1}$. We now have two representations of $C$ and can equate them together.

$$C = \frac{E[T]}{n + 1} = \frac{\beta(\frac{3}{2}, n)}{\beta(1, n)} \implies E[T] = \frac{(n + 1)\beta(\frac{3}{2}, n)}{\beta(1, n)}$$

We entirely express the answer to our problem with the $\beta$ function! Pretty neat right?