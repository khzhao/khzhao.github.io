+++
title = "On Gauss-Legendre quadrature"
date = "2026-06-07T23:58:09-05:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = []
+++

If you are familiar with the Legendre polynomials, they are a useful algebraic tool not only for solving differential equations but for other areas of mathematics, like numerical integration. Ultimately, one can arrive at the Legendre polynomials through several means. I list a few of them below:
- Directly through Gram-Schmidt of $1, x, x^2, ...$ in the Hilbert space $L^2[-1, 1]$
- Via the infamous ordinary Legendre differential equation
- Through a generating function argument with Coulomb potential

There is a relatively straightforward formula for the Legendre polynomials, which is given by the Rodrigues formula:
$$P_l(x) = \frac{1}{2^l l!} \frac{d^l}{dx^l} (x^2 - 1)^l$$

A useful property of the Legendre polynomials, if we interpret them through the Gram-Schmidt process, is that for $m < n$:
$$\langle P_m, P_n \rangle = \int_{-1}^1 P_m(x) P_n(x) dx = 0$$

Since this post is about the Gauss-Legendre quadrature, let's observe the following setting where we are interested in integrating a $(n-1)$-degree polynomial $f(x)$ over $[-1, 1]$. Suppose we have a set of $n$ points $x_1, x_2, ..., x_n$ in $[-1, 1]$ and a set of weights $w_1, w_2, ..., w_n$. Then we can approximate the integral of $f(x)$ over $[-1, 1]$ by the following weighted sum:
$$\int_{-1}^1 f(x) dx \approx \sum_{i=1}^n w_i f(x_i)$$

Let us define the following Lagrange basis polynomial:
$$l_i(x) = \prod_{j \neq i}^n \frac{x - x_j}{x_i - x_j}$$

To find an interpolation of $f(x)$ at the points $x_1, x_2, ..., x_n$, we can use the following formula. This is valid because the Lagrange polynomials form a basis for the space of $(n-1)$-degree polynomials:
$$f(x) = \sum_{i=1}^n f(x_i) l_i(x) \implies \int_{-1}^1 f(x) dx = \sum_{i=1}^n f(x_i) \int_{-1}^1 l_i(x) dx = \sum_{i=1}^n w_i f(x_i)$$
where we define $w_i = \int_{-1}^1 l_i(x) dx$. This proves that any $(n-1)$-degree polynomial can be exactly integrated using Lagrange basis polynomials. The special thing about Gauss-Legendre is that we can choose the points $x_1, x_2, ..., x_n$ to be the roots of the $n$-degree Legendre polynomial $P_n(x)$. Now take any $(2n - 1)$-degree polynomial $f(x)$ and let $f(x) = q(x) P_n(x) + r(x)$. Then, we have that:

$$\begin{align*}
\int_{-1}^1 f(x) dx &= \int_{-1}^1 (q(x) P_n(x) + r(x)) dx \\
&= \int_{-1}^1 q(x) P_n(x) dx + \int_{-1}^1 r(x) dx \\
&= \int_{-1}^1 r(x) dx \\
&= \sum_{i=1}^n w_i r(x_i) \\
&= \sum_{i=1}^n w_i (q(x_i) P_n(x_i) + r(x_i)) \\
&= \sum_{i=1}^n w_i f(x_i)
\end{align*}$$

That's it! We've shown that any $(2n - 1)$-degree polynomial can be exactly integrated using Gauss-Legendre quadrature.
