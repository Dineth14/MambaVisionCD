# MambaVision Mathematical Notes

These notes document the analysis layer added to this fork. They focus on the MambaVision-T style hierarchy, but the equations apply to the other model scales with different channel widths and depths.

## 1. MambaVision Architecture

MambaVision is a hierarchical vision backbone. A 2D image

$$
X \in \mathbb{R}^{B \times 3 \times H \times W}
$$

is first embedded by strided convolutions into a lower-resolution feature map. The network then applies four stages:

$$
\{F_1, F_2, F_3, F_4\} = \operatorname{MambaVision}(X),
$$

where early stages emphasize local convolutional features and later stages mix tokens with Mamba-style SSM blocks plus self-attention.

For a spatial feature map with channel dimension \(C\), the transformer-style stages flatten local windows into token sequences

$$
X_w \in \mathbb{R}^{B_w \times L \times C},
$$

where \(L = w^2\) for a window of side length \(w\).

### 1.1 Mixer Block

The MambaVision Mixer begins with normalization and a projection:

$$
\tilde{X} = \operatorname{LN}(X), \qquad [X_s, X_z] = \operatorname{split}(\tilde{X}W_{\text{in}}).
$$

The two branches are:

$$
U_s = \operatorname{SiLU}(\operatorname{Conv1D}_{\text{regular}}(X_s)),
$$

$$
U_z = \operatorname{SiLU}(\operatorname{Conv1D}_{\text{regular}}(X_z)).
$$

The first branch is passed to a selective state-space model:

$$
Y_s = \operatorname{SelectiveScan}(U_s; A, B(U_s), C(U_s), \Delta(U_s), D).
$$

The second branch is a symmetric non-SSM path:

$$
Y_z = U_z.
$$

Fusion is concatenation followed by a projection:

$$
Y = [Y_s; Y_z]W_{\text{out}}.
$$

The residual block is:

$$
X' = X + \operatorname{DropPath}(\gamma_1 \cdot \operatorname{Mixer}(\operatorname{LN}(X))),
$$

$$
X_{\text{out}} = X' + \operatorname{DropPath}(\gamma_2 \cdot \operatorname{MLP}(\operatorname{LN}(X'))).
$$

### 1.2 Continuous-to-Discrete SSM

The continuous state-space model is:

$$
h'(t) = Ah(t) + Bx(t),
$$

$$
y(t) = Ch(t) + Dx(t).
$$

For a zero-order hold over time step \(\Delta\), the exact discretization is:

$$
\bar{A} = \exp(\Delta A),
$$

$$
\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I)\Delta B.
$$

The recurrence becomes:

$$
h_k = \bar{A}h_{k-1} + \bar{B}x_k,
$$

$$
y_k = Ch_k + Dx_k.
$$

MambaVision uses the Mamba convention of parameterizing stable continuous dynamics with

$$
A = -\exp(A_{\log}),
$$

which encourages negative real continuous-time eigenvalues. With \(\Delta > 0\), the discrete transition satisfies

$$
|\lambda(\bar{A})| = |\exp(\Delta \lambda(A))| < 1
$$

when \(\lambda(A)\) has negative real part.

### 1.3 Selective Scan

In a fixed SSM, \(B\), \(C\), and \(\Delta\) are constants. In a selective SSM, they are functions of the current token sequence:

$$
[\Delta_k, B_k, C_k] = W_x x_k.
$$

Thus, two different inputs induce two different recurrences:

$$
h_k^{(a)} = \bar{A}_k^{(a)}h_{k-1}^{(a)} + \bar{B}_k^{(a)}x_k^{(a)},
$$

$$
h_k^{(b)} = \bar{A}_k^{(b)}h_{k-1}^{(b)} + \bar{B}_k^{(b)}x_k^{(b)}.
$$

This is the mathematical source of content-adaptive scanning.

### 1.4 Symmetric Non-SSM Branch

Causal sequence models are naturally directional. In vision, this can be limiting because spatial context is not intrinsically left-to-right. MambaVision replaces the causal convolution used in language Mamba with regular convolution in the visual mixer. The symmetric branch is:

$$
Y_z = \operatorname{SiLU}(\operatorname{Conv1D}_{\text{same}}(X_z)).
$$

Because the convolution uses same-padding over the local token sequence, output token \(t\) may depend on \(t-1\), \(t\), and \(t+1\):

$$
Y_z[t] = \sum_{r=-k}^{k} W_r X_z[t+r].
$$

This compensates for directional bias and supplies local non-causal evidence.

## 2. Why This Architecture Works

### 2.1 Regular Convolution in Vision

Language modeling uses causal convolution so future tokens cannot leak into next-token prediction. Image recognition has no such causality constraint. A patch can use evidence from all neighboring patches. Regular convolution therefore fits the spatial domain better:

$$
\operatorname{Conv}_{\text{causal}}(x_t) = \sum_{r \le 0} W_r x_{t+r},
$$

$$
\operatorname{Conv}_{\text{regular}}(x_t) = \sum_{-m \le r \le m} W_r x_{t+r}.
$$

### 2.2 Attention Only in Final Stages

Self-attention has quadratic token complexity:

$$
\operatorname{cost}_{\text{attn}} = O(N^2C),
$$

while SSM scanning is linear in token count:

$$
\operatorname{cost}_{\text{SSM}} = O(NC).
$$

At high spatial resolution, \(N\) is large, so attention is expensive. At low resolution, such as \(14 \times 14\) and \(7 \times 7\), attention becomes affordable and useful for global semantic interactions. MambaVision therefore places self-attention in later stages, where \(N\) is small and the features are semantically richer.

### 2.3 Complexity Summary

For \(N\) tokens, channel width \(C\), and SSM state size \(S\):

$$
\operatorname{SSM}(N, C, S) \approx O(NCS),
$$

$$
\operatorname{Attention}(N, C) \approx O(N^2C).
$$

When \(N\) is large, the SSM branch provides long-range mixing without quadratic memory growth.

## 3. Relationship to Prior Work

### 3.1 VMamba vs. MambaVision

VMamba uses visual state-space scanning as the central token mixer. MambaVision is hybrid: it keeps convolutional early stages, uses MambaVision Mixer blocks in later stages, and adds self-attention where resolution is low. The key distinction is that MambaVision explicitly includes a symmetric non-SSM branch and attention in the hierarchy.

### 3.2 Vision Mamba (Vim) vs. MambaVision

Vision Mamba frames images as token sequences and uses bidirectional sequence modeling to reduce directional bias. MambaVision instead combines selective SSM scanning with a non-causal convolutional branch and late-stage attention. Vim emphasizes sequence symmetry; MambaVision emphasizes hierarchical visual efficiency.

### 3.3 Swin Transformer vs. MambaVision

Swin uses shifted-window attention:

$$
\operatorname{Attn}(Q, K, V) = \operatorname{softmax}(QK^\top / \sqrt{d})V.
$$

MambaVision uses window tokenization in the later stages too, but many blocks use linear SSM mixing rather than attention. The result is a hybrid between ConvNet locality, SSM sequence filtering, and transformer-style semantic aggregation.

## 4. Change Detection Relevance

Change detection compares two images \(X^{(t_1)}\) and \(X^{(t_2)}\). A Siamese MambaVision encoder can share weights:

$$
\{F_i^{(t_1)}\}_{i=1}^4 = E_\theta(X^{(t_1)}),
$$

$$
\{F_i^{(t_2)}\}_{i=1}^4 = E_\theta(X^{(t_2)}).
$$

Early features \(F_1, F_2\) preserve edges, textures, roads, roofs, field boundaries, and small geometric shifts. Later features \(F_3, F_4\) carry broader semantic context through SSM and attention.

### 4.1 Proposed Cross-Temporal Fusion

A simple fusion tensor at stage \(i\) is:

$$
Z_i = \operatorname{Concat}(F_i^{(t_1)}, F_i^{(t_2)}, |F_i^{(t_2)} - F_i^{(t_1)}|).
$$

A stronger cross-temporal attention fusion is:

$$
\operatorname{CTA}(F_1, F_2) =
\operatorname{softmax}\left(\frac{Q(F_1)K(F_2)^\top}{\sqrt{d}}\right)V(F_2).
$$

For change detection, useful insertion points are:

- Stage 1 or 2 for boundary-sensitive changes.
- Stage 3 for object-level semantic changes while retaining \(14 \times 14\) spatial structure.
- Multi-scale fusion across all stages for dense prediction.

The open research opportunity is to preserve MambaVision's efficient long-range modeling while adding explicit bitemporal interaction rather than only subtracting Siamese features after encoding.
