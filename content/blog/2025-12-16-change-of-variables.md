+++
title = "Changing variables" 
date = "2025-12-16T23:09:55-06:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = []
+++

This post will be a bit more basic than usual but is quite fundamental for understanding **core** concepts in probabilty and statistics. Learning how to change variables correctly and also understanding the meaning of the Jacobian is crucial. The applications of these are everywhere especially in deep learning. 

Let's go back to basics. One of the most basic concepts that we are concerned with is analyzing rate of change. If we have a relation $y = f(x)$, we are interested in analyzing how $y$ changes with respect to $x$. This is the rate of change of $y$ with respect to $x$ and is given by the derivative:
$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$ 

In differential terms, we can write $dy = f'(x) dx$. Now this concept is pretty straightforward in the one-dimensional case. If we increase the number of dimensions, the concept is harder to relate. Suppose that we have:

\begin{align*}
x = f(u, v, w) \\\
y = g(u, v, w) \\\
z = h(u, v, w)
\end{align*}

We are interested in analyzing how $x, y, z$ change with respect to $u, v, w$. We can write the total differential of $x, y, z$ as:
\begin{align*}
dx = \frac{\partial x}{\partial u} du + \frac{\partial x}{\partial v} dv + \frac{\partial x}{\partial w} dw \\\
dy = \frac{\partial y}{\partial u} du + \frac{\partial y}{\partial v} dv + \frac{\partial y}{\partial w} dw \\\
dz = \frac{\partial z}{\partial u} du + \frac{\partial z}{\partial v} dv + \frac{\partial z}{\partial w} dw
\end{align*}

In matrix form we can write this as:
\begin{align*}
\begin{bmatrix}
dx \\\
dy \\\
dz
\end{bmatrix} = \begin{bmatrix}
\frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} & \frac{\partial x}{\partial w} \\\
\frac{\partial y}{\partial u} & \frac{\partial y}{\partial v} & \frac{\partial y}{\partial w} \\\
\frac{\partial z}{\partial u} & \frac{\partial z}{\partial v} & \frac{\partial z}{\partial w} 
\end{bmatrix} \begin{bmatrix} du \\\ dv \\\ dw \end{bmatrix}
\end{align*}

Under this transformation, if we look at the vector $(du, 0, 0)$ and apply the transformation, we get:
\begin{align*}
\begin{bmatrix}
dx \\\
dy \\\
dz
\end{bmatrix} = \begin{bmatrix}
\frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} & \frac{\partial x}{\partial w} \\\
\frac{\partial y}{\partial u} & \frac{\partial y}{\partial v} & \frac{\partial y}{\partial w} \\\
\frac{\partial z}{\partial u} & \frac{\partial z}{\partial v} & \frac{\partial z}{\partial w} 
\end{bmatrix} \begin{bmatrix} du \\\ 0 \\\ 0 \end{bmatrix} = \begin{bmatrix}
\frac{\partial x}{\partial u} du \\\
\frac{\partial y}{\partial u} du \\\
\frac{\partial z}{\partial u} du
\end{bmatrix}
\end{align*}

Similarly, if we look at the vector $(0, dv, 0)$ and apply the transformation, we get:
\begin{align*}
\begin{bmatrix}
dx \\\
dy \\\
dz
\end{bmatrix} = \begin{bmatrix}
\frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} & \frac{\partial x}{\partial w} \\\
\frac{\partial y}{\partial u} & \frac{\partial y}{\partial v} & \frac{\partial y}{\partial w} \\\
\frac{\partial z}{\partial u} & \frac{\partial z}{\partial v} & \frac{\partial z}{\partial w} 
\end{bmatrix} \begin{bmatrix} 0 \\\ dv \\\ 0 \end{bmatrix} = \begin{bmatrix}
\frac{\partial x}{\partial v} dv \\\
\frac{\partial y}{\partial v} dv \\\
\frac{\partial z}{\partial v} dv
\end{bmatrix}
\end{align*}

And lastly, if we look at the vector $(0, 0, dw)$ and apply the transformation, we get:
\begin{align*}
\begin{bmatrix}
dx \\\
dy \\\
dz
\end{bmatrix} = \begin{bmatrix}
\frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} & \frac{\partial x}{\partial w} \\\
\frac{\partial y}{\partial u} & \frac{\partial y}{\partial v} & \frac{\partial y}{\partial w} \\\
\frac{\partial z}{\partial u} & \frac{\partial z}{\partial v} & \frac{\partial z}{\partial w} 
\end{bmatrix} \begin{bmatrix} 0 \\\ 0 \\\ dw \end{bmatrix} = \begin{bmatrix}
\frac{\partial x}{\partial w} dw \\\
\frac{\partial y}{\partial w} dw \\\
\frac{\partial z}{\partial w} dw
\end{bmatrix}
\end{align*}

Now we turn to the integration problem, to see this more clearly. Suppose that we have the following Riemann sum and let $P_i$ be a partition of $R$ into $n$ subregions.
$$\sum_{i=1}^n f(\vec{x_i}) \text{Area}(P_i)$$
where $\vec{x_i}$ is a point in the $i$-th subregion and $\text{Area}(P_i)$ is the area of the $i$-th subregion. We are interested in changing variables to a new variable $\vec{x} = g(\vec{y})$. The new Riemann sum is:
$$\sum_{i=1}^n f(g(\vec{y_i})) \text{Area}(P_i)$$
Let $S_i = g^{-1}(P_i)$ and $y_i \in S_i$. Then, we have that that the above is equal to:
$$\sum_{i=1}^n f(g(\vec{y_i})) \text{Area}(g(S_i))$$
By the Jacobian scaling, we know that:
$$Area(P_i) \approx |\det(J)| \text{Area}(S_i)$$
which implies the above to be equal to:
$$\sum_{i=1}^n f(g(\vec{y_i})) |\det(J)| \text{Area}(S_i)$$
As we let $n \to \infty$ and the subregions get smaller and smaller, we get:
$$\int_{g^{-1}(R)} f(g(\vec{y})) |\det(J)| d\vec{y}$$
which concludes this hand-wavy proof.