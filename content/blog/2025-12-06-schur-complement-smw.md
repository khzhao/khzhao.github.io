+++
title = "Schur! Let's invert it!"
date = "2025-12-06T19:15:44-06:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = []
+++

When I was an undergraduate, there were lots of identities that I just memorized without paying much attention to how they were derived. Sometimes this is completely fine especially if they are not critical for understanding **core** concepts. One result that contradicted this philosophy was the Sherman-Morrison-Woodbury identity. The identity can be simply stated as follows:
\begin{align*}
(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + V A^{-1}U)^{-1}V A^{-1}
\end{align*}

where $A^{-1}, C^{-1}$ are known and $U, V$ are low-rank matrices. You may be more familiar with a special case of the identity for rank-1 updates:
$$(I + uv^T)^{-1} = I - \frac{uv^T}{1 + v^T u}$$

Learning how to derive this is quite useful for intuition. The identities come up naturally when computing results for block inversion. We are given $A^{-1}$ and we are concerned with $(A + uv^T)^{-1}$. Let us consider the following equation:
$$(A + uv^T)x = b$$

We note that for general block matrix inversion where $S = A - BD^{-1}C$:
$$\begin{bmatrix} A & B \\\ C & D \end{bmatrix}^{-1} = \begin{bmatrix} S^{-1} & -S^{-1}BD^{-1} \\\ -D^{-1}CS^{-1} & D^{-1} + D^{-1}CS^{-1}BD^{-1} \end{bmatrix}$$

For our earlier matrix, the Schur complement of the bottom right $I$ matrix is $A + uv^T$. The above identity shows that if we can invert the Schur complement then inverting a block matrix is straightforward. Now let's prove the identity:
$$x = A^{-1}(b - uv^Tx)$$
Let $s = v^Tx$. Then, we have that:
$$s = v^T(A^{-1}b - A^{-1}us)$$
After simplifying, we get:
$$(1 + v^TA^{-1}u)s = v^TA^{-1}b \implies s = \frac{v^TA^{-1}b}{1 + v^TA^{-1}u}$$
This implies that the final equation is:
$$x = A^{-1}(b - u\frac{v^TA^{-1}b}{1 + v^TA^{-1}u}) = A^{-1}(b - \frac{uv^TA^{-1}b}{1 + v^TA^{-1}u})$$
which gives us:
$$x = (A^{-1} - \frac{A^{-1}uv^TA^{-1}}{1 + v^TA^{-1}u})b$$
This implies that:
$$(A + uv^T)^{-1} = A^{-1} - \frac{A^{-1}uv^TA^{-1}}{1 + v^TA^{-1}u}$$
which is the Sherman-Morrison-Woodbury identity!

An exercise to the reader is to derive the matrix version of the Sherman-Morrison-Woodbury identity:
$$(A + UCV)^{-1}$$
*Hint*: Use the block matrix inversion formula and compute the Schur complements of the top left and bottom right blocks. Represent the matrix in both forms using both Schur complements.
