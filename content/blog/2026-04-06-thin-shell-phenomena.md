+++
title = "Thin shell phenomena for n-balls"
date = "2026-04-06T17:30:18-05:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = []
+++

This is a classic result but often overlooked and helps motivate why curse of dimensionality is such an annoying problem. Why do we have this curse of dimensionality? A lot of the intuition comes from understanding the ratio between the volumes of the unit sphere and the unit cube. As we know, since:

$$Vol(B_n) = \frac{\pi^{n/2}}{\Gamma(n/2 + 1)} \rightarrow_{n \to \infty} 0$$
$$\implies \frac{Vol(B_n)}{Vol(C_n)} = \frac{\pi^{n/2}}{\Gamma(n/2 + 1)} \rightarrow_{n \to \infty} 0$$

Interestingly, the corners of the unit cube become exponentially further in distance as we increase the dimension. It makes sense we need more bit flips when traversing the edges to get there, at least in terms of Manhattan distance. But more curiously, we have another phenomena for n-balls, which is that almost the entirety of the mass is concentrated within a thin shell. To see this, we first look at the concentration of the norm. Suppose we have an isotropic vector $X$, then:
$$E||x||_2^2 = n$$
By Jensen's inequality, we know that:
$$E||X||_2 \leq \sqrt{n}$$
So off the bat, we know that the norm is in expectation bounded by $\sqrt{n}$. But actually, there's a more powerful result which says that actually the L2 norm concentrates around $\sqrt{n}$! Note that if $X$ is a sub-gaussian vector and isotropic, then $||X||_2^2$ is an sub-exponential random variable! We can apply Bernstein's inequality and show that the norm follows a sub-gaussian concentration around $\sqrt{n}$! This means the norm $\approx O(\sqrt{n})$ with high probability and because there is a sub-gaussian concentration bound, we know that:

$$E[||X||\_2 - \sqrt{n}] \leq O(C||X||_{\psi_2})$$

where $||X||_{\psi_2}$ is the $\psi_2$ norm of $X$.