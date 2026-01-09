+++
title = "A neat result regarding log-sum-exp"
date = "2026-01-08T18:07:25-06:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = []
+++

Here is a neat proof regarding the log-sum-exp function that shows that it is a convex function.

**Claim**: $f(x) = \log \sum_{i=1}^n e^{x_i}$ is a convex function.

**Proof**: We can show that the Hessian is positive semi-definite. The gradient is given by:
$$\nabla f(x)\_i = \frac{e^{x\_i}}{\sum_{j=1}^n e^{x\_j}}$$
This means that the Hessian is given by:
$$\nabla^2 f(x)\_{i, j} = \begin{cases} \frac{e^{x\_i}}{\sum_{j=1}^n e^{x\_j}} - \frac{e^{x_i}e^{x_i}}{(\sum\_{j=1}^n e^{x_j})^2} & i = j \\\ -\frac{e^{x_i}e^{x_j}}{(\sum\_{j=1}^n e^{x_j})^2} & i \neq j \end{cases}$$

We let $S(x) = \nabla f(x)$, then we can represent the Hessian as:
$$H(f) = \text{diag}(S(x)) - S(x)S(x)^T$$

Note that because the softmax function is normalized to 1, we can employ Jensen's inequality to show that the Hessian is positive semi-definite. We are concerned with:
$$x^T H(f) x = x^T (\text{diag}(S(x)) - S(x)S(x)^T) x $$

However, we have that:
$$x^T S(x)S(x)^T x = (\sum_{j=1}^n S_j x_j)^2 \leq \sum_{j=1}^n S_j x_j^2 = x^T \text{diag}(S(x)) x$$

This implies that: $$x^T H(f) x \geq 0$$

Thus, we are done with this proof. Additionally, there is a neat relation to the $\max$ function. Let us observe that:
$$\max ( e\^{x_1}, \dots, e^{x_n})  \leq e^{x_1} + \dots + e^{x_n} \leq n \max (e^{x_1}, \dots, e^{x_n} )$$
Because $\log$ is a monotonically increasing function, we have that:
$$\log(\max ( e\^{x_1}, \dots, e^{x_n})) \leq \log(e^{x_1} + \dots + e^{x_n}) \leq \log(n \max (e^{x_1}, \dots, e^{x_n} ))$$
This implies that:
$$\max (x_1, \dots, x_n) \leq \log(\sum_{i=1}^n e^{x_i}) \leq \max (x_1, \dots, x_n) + \log(n)$$
