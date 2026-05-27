# Mathematical Analysis: DEPN — Detecting and Editing Privacy Neurons

> **Paper**: *"DEPN: Detecting and Editing Privacy Neurons in Pretrained Language Models"*, EMNLP 2023, arXiv:2310.20138  
> **File**: `depn-algo.ipynb`  
> **Result**: avg_toxicity 0.2147 → **0.1548** | toxic_ratio 15.5% → **12.0%** | PPL 13.88 → **14.34**

---

## 1. What Is a "Toxic Neuron"?

DEPN proposes that specific neurons in the FFN layers of a transformer are disproportionately responsible for generating harmful content. Identifying and zeroing these neurons provides a **training-free**, **interpretable** unlearning mechanism.

### OPT-1.3B FFN Architecture

Each of OPT-1.3B's 24 decoder layers contains a two-layer FFN:

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$$

where:
- $W_1 \in \mathbb{R}^{8192 \times 2048}$ (`fc1`) — projects from hidden dim to FFN dim
- $W_2 \in \mathbb{R}^{2048 \times 8192}$ (`fc2`) — projects back
- **Neuron $k$ at layer $l$** = the $k$-th unit of $h_l = \text{ReLU}(W_1 x + b_1)$, i.e., $h_l^k = \text{ReLU}((W_1)_{k,:} x + (b_1)_k)$

Total neurons: $24 \times 8192 = \mathbf{196,608}$.

---

## 2. Attribution Score: Gradient × Activation

DEPN scores each neuron $(l, k)$ using an **integrated gradients** approximation. The exact formulation (Eq. 3 in the paper) is:

$$\text{Att}(w_l^k) = \beta_l^k \int_0^{\beta_l^k} \frac{\partial P(\mathbf{Y} \mid \mathbf{X}, \alpha_l^k)}{\partial w_l^k} d\alpha_l^k$$

where $\beta_l^k$ is the current activation of neuron $(l, k)$ and $\alpha_l^k$ is the interpolated activation along the integration path.

For $m=1$ (single-step approximation used in practice):

$$\boxed{\text{Att}(w_l^k) \approx h_l^k \cdot \left(-\frac{\partial \mathcal{L}_{\text{NLL}}}{\partial h_l^k}\right)}$$

where $h_l^k = \text{ReLU}((W_1)_{k,:} x + (b_1)_k) \geq 0$ is the neuron activation and $-\partial \mathcal{L}_{\text{NLL}} / \partial h_l^k$ is the negative gradient of the NLL loss with respect to that activation.

### Interpretation of the score

A positive score $\text{Att}(w_l^k) > 0$ requires **both**:

1. $h_l^k > 0$ — the neuron is **active** (fires non-zero)
2. $-\partial \mathcal{L}_{\text{NLL}} / \partial h_l^k > 0$ — **increasing** this neuron's activation would **increase** the model's probability of toxic tokens (decrease NLL)

In other words: neuron $(l, k)$ is "toxic" if and only if it fires on toxic inputs AND its activation promotes toxic token generation.

The score is **clamped at 0**:

$$\text{Att}(w_l^k) = \max\left(0,\; h_l^k \cdot \left(-\frac{\partial \mathcal{L}_{\text{NLL}}}{\partial h_l^k}\right)\right)$$

This ensures we only count neurons that actively **promote** toxicity (not neurons that fire but happen to suppress toxic tokens).

---

## 3. Hook Implementation for Efficient Attribution

Rather than backpropagating through the full model 196,608 times (once per neuron), DEPN uses a single forward+backward pass per scoring sample, with **hooks** to capture activations and gradients simultaneously.

### Hook placement

The hooks are placed on `fc2` (the second linear layer), because:

- **Forward pre-hook on `fc2`**: captures `inp[0]` = $h_l = \text{ReLU}(W_1 x + b_1)$ — this is exactly the neuron activation vector we need
- **Backward hook on `fc2`**: captures `grad_inp[0]` = $\partial \mathcal{L} / \partial h_l$ — the gradient flowing back into `fc2`, which equals the gradient w.r.t. the activation

```python
# fc2 input = ReLU(fc1(x)) = [B, T, 8192] — neuron activations
def _fwd(li):
    def hook(module, inp):
        layer_acts[li] = inp[0].detach().float()
    return hook

# fc2 grad_input = d_loss/d_activation = [B, T, 8192]
def _bwd(li):
    def hook(module, grad_inp, grad_out):
        layer_grads[li] = grad_inp[0].detach().float()
    return hook
```

This gives us, in a single backward pass:
- $h_l \in \mathbb{R}^{B \times T \times 8192}$ for all 24 layers simultaneously
- $\nabla_{h_l} \mathcal{L} \in \mathbb{R}^{B \times T \times 8192}$ for all 24 layers simultaneously

### Per-sample aggregation

For each scoring sample, the score is averaged over batch and sequence dimensions:

$$\hat{\text{Att}}(w_l^k) = \frac{1}{B \cdot T} \sum_{b=1}^{B} \sum_{t=1}^{T} \max\left(0,\; h_{l,b,t}^k \cdot (-\nabla_{h_{l,b,t}^k} \mathcal{L})\right)$$

Accumulated over $N = 64$ scoring samples:

$$\overline{\text{Att}}(w_l^k) = \frac{1}{N} \sum_{n=1}^{N} \hat{\text{Att}}_n(w_l^k)$$

---

## 4. Neuron Pruning: Permanent Weight Zeroing

The top-$Z$ neurons by score are permanently disabled by zeroing their corresponding weights:

For neuron $(l, k)$:

$$\begin{cases}
(W_2)_{:,k} \leftarrow \mathbf{0} & \text{(fc2 column $k$: neuron $k$'s contribution to output)} \\
(W_1)_{k,:} \leftarrow \mathbf{0} & \text{(fc1 row $k$: neuron $k$'s input weights)} \\
(b_1)_k \leftarrow 0 & \text{(fc1 bias for neuron $k$)}
\end{cases}$$

After zeroing, for **any** input $x$:

$$h_l^k = \text{ReLU}\left(\underbrace{(W_1)_{k,:} x + (b_1)_k}_{= 0}\right) = 0$$

$$\text{fc2 output contribution of neuron } k = (W_2)_{:,k} \cdot h_l^k = \mathbf{0}$$

The neuron is **permanently silent** — no runtime overhead, no masks, no hooks needed at inference.

---

## 5. Observed Scoring Distribution

From `log_depn.txt`:

```
Non-zero neurons : 196,608 / 196,608 (100.0%)
Mean (non-zero)  : 0.000001
Max              : 0.000003
Top-1000 neurons : ALL in Layer 23
```

### Analysis: Why all top neurons in Layer 23?

This is a well-known property of gradient-based attribution in deep networks: **gradients are larger in layers closer to the loss**. Layer 23 is the final decoder layer, immediately before the language modelling head. The gradient $\partial \mathcal{L} / \partial h_{23}^k$ is larger than $\partial \mathcal{L} / \partial h_{0}^k$ because it involves fewer chain-rule multiplications.

This creates a **gradient bias** in DEPN's attribution score: earlier layers are systematically underscored relative to their true causal importance for toxicity.

**Consequence**: all 1000 pruned neurons come from layer 23 in our run (N=64 samples). With more scoring samples or per-layer score normalisation, the distribution would spread across layers.

### Mitigation: Per-layer normalisation (not applied in this run)

To correct for gradient magnitude differences across layers, one could normalise within each layer before taking the global top-$Z$:

$$\tilde{\text{Att}}(w_l^k) = \frac{\overline{\text{Att}}(w_l^k)}{\max_k \overline{\text{Att}}(w_l^k)}$$

This ensures each layer contributes proportionally regardless of absolute gradient scale.

---

## 6. Kaggle Implementation Optimizations

### 6.1 CPU score accumulator (avoids VRAM pressure)

The full score tensor is $24 \times 8192 = 196,608$ float32 values = **800 KB**. Keeping it on CPU:

```python
scores = torch.zeros(n_layers, ffn_dim, dtype=torch.float32)  # CPU
# ...
scores[l] += attr.mean(dim=(0, 1)).cpu()  # move to CPU before accumulating
```

**Critical fix**: `.cpu()` is required. The hook tensors live on CUDA (where the model's activation was computed), but `scores` is a CPU tensor. Without `.cpu()`, PyTorch raises `RuntimeError: Expected all tensors to be on the same device`.

### 6.2 Batch size 1 for scoring

Using `SCORE_BATCH_SIZE=1` means:
- Each backward pass computes gradients for a single sample
- Weight gradients are **released immediately** with `zero_grad(set_to_none=True)`
- Peak VRAM stays at **3.19 GB** (model 2.6 GB + activations + weight grads for one sample)

With batch size > 1, gradient accumulation in the parameter `.grad` tensors would keep 2.6 GB of gradient memory alive across iterations.

### 6.3 AMP autocast during scoring

```python
with torch.cuda.amp.autocast():
    out = model(input_ids=ids, attention_mask=mask, labels=lbl)
    loss = out.loss
loss.backward()
```

AMP reduces activation memory during the forward pass. The backward pass still uses fp32 for gradient accuracy (gradients are upcast automatically by autocast).

### 6.4 `model.config.use_cache = False`

OPT's KV-cache is incompatible with gradient computation. Setting `use_cache=False` forces the model to recompute all hidden states during the backward pass, which is necessary for the backward hooks to fire correctly.

---

## 7. Comparison with DEPN Paper Theory

| Paper assumption | Our observation | Gap |
|-----------------|----------------|-----|
| Neurons distributed across layers | All 1000 in layer 23 | Gradient bias; fixed with more samples or normalisation |
| Used on BERT (12 layers, 3072 FFN) | OPT-1.3B (24 layers, 8192 FFN) | 8× more neurons; Z=1000 = 0.51% vs paper's 500/18432 = 2.7% |
| Memorised text unlearning (specific phrases) | Distributed toxicity style | DEPN is less effective for distributional shifts |
| N=200 scoring samples | N=64 (speed/memory trade-off) | Fewer samples → noisier scores → layer 23 dominance |

### Why DEPN is less effective for toxicity (28% reduction vs NPO's 99.6%)

DEPN was designed for **memorised specific information** (e.g., "SSN: 123-45-6789"). Toxicity is a **distributional style property** — it is encoded across many neurons and layers, not concentrated in a small set. Zeroing 1000 neurons (0.51%) removes some toxic capacity but leaves the majority intact.

For comparison, the original DEPN paper achieves near-complete memorisation removal with 200–500 neurons on BERT because the target is specific token sequences with high localisation.
