+++
title = "[draft] Representations of memory"
date = "2026-05-25T18:26:09-05:00"
toc = true

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = []
+++

The transformer architecture has been KING for the past decade. Nobody has been able to beat it yet. There have been valiant attempts (MAMBAv1-3, SSMs, etc), but they have all failed to show their benefits at large scales. That being said, it is still useful to understand the tradeoffs of these techniques. Qwen3.5 and Kimi2.5 have incorporated recently "hybrid" architectures, mixing softmax attention with regular delta nets. Let's talk about this.  

### State space models

In continuous time, we often model state space models as the evolution of a latent/hidden state or system. It is represented as:
$$\frac{d h(t)}{dt} = f(h(t), x(t), t)$$
where $h(t)$ is the hidden state, $x(t)$ is the input, and $f$ is a function that defines the evolution of the hidden state.

For simplicity, we can restrict ourselves to the linear dynamical system scenario, where:
$$\frac{d h(t)}{dt} = A h(t) + B x(t)$$
$$y(t) = C h(t) + D x(t)$$

In the context of autoregressive modeling, we can think of the hidden state as the "memory", and the equation for $y(t)$ as the reading operation from memory. In this case, we actually have sort of a closed form solution for the hidden state, which we derive below.

**Lemma**: The solution to the linear dynamical system is given by:
$$h(t) = e^{At} h(0) + \int_0^t e^{A(t - s)} B x(s) ds$$

**Proof**: We can solve this by using the method of integrating factors. We have that:
$$\begin{align*}
\frac{d}{ds} (e^{-As} h(s)) &= -A e^{-As} h(s) + e^{-As} (A h(s) + B x(s)) \\
&= e^{-As} B x(s)
\end{align*}$$

Integrating both sides, we get:
$$\begin{align*}
\int_0^t \frac{d}{ds} (e^{-As} h(s)) ds &= \int_0^t e^{-As} B x(s) ds \\
e^{-At} h(t) - h(0) &= \int_0^t e^{-As} B x(s) ds \\
h(t) &= e^{At} h(0) + B \int_0^t e^{A(t - s)} x(s) ds
\end{align*}$$

This is the "sort-of" closed form solution for the hidden state. For systems with a dynamic $B(t)$, then we get:
$$h(t) = e^{At} h(0) + \int_0^t e^{A(t - s)} B(s) x(s) ds$$

However, what we actually observe in data is $x_1, x_2, \dots, x_t$ and we are concerned with the hidden state $h_1, h_2, \dots, h_t$. Our formula holds in a continuous regime. We can generalize our formula to the discrete regime by invoking the "zero-order hold" condition, which states that $x(t) = x_t$ for all $t \in [t_i, t_{i+1})$. This means that:

$$h_{t+\delta} = e^{A\delta}h(t) + \left(\int_t^{t+\delta} e^{A(t+\delta - r)} B(r) dr\right) x(t+\delta)$$

In a usual discrete dynamical system, we are given the system:
$$h_{t+1} = A_d h_t + B_d x_t$$
$$y_t = C_d h_t + D_d x_t$$

If we let $A_d = e^{A\delta}$ and $B_d = \int_t^{t+\delta} e^{A(t+\delta - r)} B(r) dr$, then we get exactly the discretized linear dynamical system.

### A model for memory

In regular transformers, we can represent the hidden state of a block as $H_t^l = (K_{1:t}^l, V_{1:t}^l)$. This is in other words known as the KV-cache (for better or worse). It allows us to retrieve facts or knowledge that the model has learned to represent at layer $l$. We are often given a query vector $q_t$, where $t$ denotes the time step (more accurately the $t$-th token in the sequence). For ordinary softmax attention, we obtain the new representation of the token via the operation:
$$o_t = \sum_{i=1}^t \text{softmax}\left(\frac{q_t k_i^T}{\sqrt{d_k}}\right)v_i$$

This is certainly one model for memory (and it is quite strong), but the memory cost of this is quite high. $K$ and $V$ are both $t \times d_k$ matrices, so the memory cost is $O(t d_k)$. This is quite expensive for long sequences. If we could somehow reduce the memory cost AND preserve the experimental performance of softmax attention, then that would be super ideal. 

#### Linear attention

If we squint our eyes a little, we can see that softmax attention is merely retrieving value vectors from memory based on how similar the query is to the keys. But ultimately it is a convolution over the key vocabulary. In other words, as long as we can define a similarity metric, then we can perform something similar. One annoying part of the softmax attention is that we need to keep track of the entire $t \times t$ matrix (though it is usually not fully materialized). An ideal form of attention would be something like: 
$$o_t = \sum_{i=1}^t \phi(q_t) \phi(k_i)^T v_i$$
where $\phi$ is some kernel function. We can reduce **immensely** the computation load of attention if it is merely just a series of dot products. But kernel theory has been quite underwhelming as of late, so let's just use a regular identity kernel. We observe that our attention operation can be:
$$o_t = \sum_{i=1}^t q_t k_i^T v_i = q_t \sum_{i=1}^t k_i^T v_i$$
We see that the inner sum is a sum of outer products and that there exists a recurrent relation. Denote $S_n = \sum_{i=1}^n k_i v_i$, then we have that:
$$S_{n+1} = S_n + k_{n+1}^T v_{n+1} \in \mathbb{R}^{d_k \times d_k}$$
We can write:
$$o_t = q_t S_t = q_t (S_{t-1} + k_t^T v_t) = q_t S_{t-1} + q_t k_t^T v_t$$
In other words, it's the memory retrieved from the most recent step corrected by the $q_t k_t^T v_t$ term. Thus, per block layer, we can associate the $S_t^l$ as the hidden state for that layer.

#### RetNet

RetNet applies a very natural concept to linear attention. In the above, formulation we are unable to "forget" information that is no longer relevant or very far back from the current time step. RetNet solves this by applying a time-dependent decay factor $\alpha(t) \in (0, 1)$ to the memory update. This means that:
$$S_t = \alpha(t) S_{t-1} + k_t^T v_t$$

#### Gated linear attention

GLA aims to solve the same problem as RetNet but through a different lens. Instead of applying a blanket decay factor, we can train a gate that takes in the current token to determine coordinate-wise how much to decay and update the memory. This means the update is:
$$S_t = G \odot S_{t-1} + k_t^T v_t$$

This gate $G$ can be a parameter-valued matrix or a something that takes in the query vector $q_t$ to determine the gate, like a SwiGLU gate for example.

#### Delta nets

For delta nets, we can view the memory as a form of online regression, where our objective is to predict the value vector using $k_t$. Let's look at the update for linear attention again:
$$S_t = S_{t-1} + k_t^T v_t$$
Let's define $\hat{v}_t = k_t^T S_{t-1}$. We can view this as the regression of $k_t$ onto $v_t$ in an online fashion. Then we update it via the prediction error:
$$S_t = S_{t-1} + \beta_t k_t^T(v_t - \hat{v}_t)$$ 
Equivalently, we have:
$$S_t = (I - \beta_t k_t^T k_t) S_{t-1} + \beta_t k_t^T v_t$$
where $\beta_t$ is a learned writing strength. The connection to online training is more evident if we see that at every time step, we obtain a pair of training examples $(k_t, v_t)$. We can then see that we are trying to learn an association $S_t$ such that $k_tS_t \approx v_t$. We are essentially solving the following optimization problem:
$$L(S) = \frac{1}{2} \min_{S} \sum_{i=1}^t (k_i S - v_i)^2$$
If we use gradient descent to solve this, let's look at the update rule:
$$\nabla_S L(S) = \sum_{i=1}^t k_i^T (k_i S - v_i)$$
$$S \leftarrow S - \eta \sum_{i=1}^t k_i^T (k_i S - v_i) = S + \eta \sum_{i=1}^t k_i^T (v_i - k_i S)$$
which is exactly the delta update rule. Another advantage of the DeltaNet compared to regular linear attention is that if the query is very similar to two keys, then their value vectors will clash. In particular for $k_i$ and $k_j$, if we have $q_t \cdot k_i \approx q_t \cdot k_j$, then $v_i$ and $v_j$ will clash. DeltaNet somewhat avoids this via the erase then write memory paradigm. 

#### Gated delta nets and generalized householder

The above formulation of the delta net which gives the update:
$$S_t = (I - \beta_t k_t^T k_t) S_{t-1} + \beta_t k_t^T v_t$$
can be viewed as a generalized householder transformation. In particular, we can write:
$$S_t = H_t S_{t-1} + \beta_t k_t^T v_t$$
where $H_t$ is a householder or projection matrix. We can generalize the concept of the gating to also the delta net as well with the update:
$$S_t = G \odot S_{t-1} + \beta k_t^T(v_t - k_t (G \odot S_{t-1}))$$

The erase and write operation need not be at the same strength level $\beta_t$. Decoupling them brings us Gated DeltaNet 2, giving us the update:
$$S_t = G \odot S_{t-1} - (w_{t_1} \odot k_t)^Tk_t (G \odot S_{t-1}) + \beta_t k_t^T (w_{t_2} \odot v_t)$$

where $w_{t_1}$ is the key-sided learned gate and $w_{t_2}$ is the value-sided learned gate. We can also view a similar update rule as:
$$S_t = G \odot S_{t-1} + (w_{t_1} \odot k_t)^T (w_{t_2} \odot (v_t - k_t (G \odot S_{t-1})))$$

#### Kimi delta attention

This is a particular instantiation of the gated delta net where the gate is a diagonal matrix. In particular, we have $D_t = \text{diag}(\alpha_t)$. We can write the update as:
$$S_t = D_t \odot S_{t-1} + \beta_t k_t^T(v_t - k_t (D_t \odot S_{t-1}))$$
where $\odot$ is the Hadamard product.

### MAMBA

Taking the above inspiration, we can construct a new SSM-based architecture, without as hard "memory" requirements as regular transformers with softmax attention. 