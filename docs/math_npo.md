# Mathematical Analysis: NPO — Negative Preference Optimization

> **Paper**: *"Negative Preference Optimization: How to Make LLMs Forget"*, arXiv:2404.05868  
> **File**: `npo-algo.ipynb`  
> **Result**: avg_toxicity 0.1591 → **0.0007** | toxic_ratio 11.5% → **0.0%** | PPL 13.88 → **14.78**

---

## 1. From DPO to NPO: Probabilistic Forgetting

### 1.1 Background: Direct Preference Optimization (DPO)

DPO (Rafailov et al., 2023) is an RLHF method that trains a model to prefer "chosen" completions over "rejected" ones without a separate reward model. The DPO loss is:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

where $y_w$ is the preferred (winning) completion and $y_l$ is the rejected (losing) one.

### 1.2 The NPO Insight

For machine unlearning, we want to **discourage** the model from generating forget-set completions. There are no "preferred" alternatives — we simply want to move the model *away* from $\mathcal{D}_f$.

NPO achieves this by treating $\mathcal{D}_f$ as the **rejected** completions and letting the "chosen" completion be implicitly defined by the reference model (the pre-trained $\pi_{\text{ref}}$). Setting $y_w = x_f$ (same as rejected, i.e., the reference just re-generates it) and $y_l = x_f$:

$$\mathcal{L}_{\text{NPO}} = -\frac{2}{\beta} \mathbb{E}_{x_f \sim \mathcal{D}_f}\!\left[\log \sigma\!\left(-\beta \log \frac{\pi_\theta(x_f)}{\pi_{\text{ref}}(x_f)}\right)\right]$$

Expanding the log-ratio:

$$\log \frac{\pi_\theta(x_f)}{\pi_{\text{ref}}(x_f)} = \sum_{t=1}^{T} \log \frac{\pi_\theta(x_{f,t} \mid x_{f,<t})}{\pi_{\text{ref}}(x_{f,t} \mid x_{f,<t})}$$

This is the **per-sequence log-probability ratio** — a single scalar measuring how much the current model $\pi_\theta$ has moved away from $\pi_{\text{ref}}$ on the forget sequence.

---

## 2. Mathematical Properties of the NPO Loss

### 2.1 Gradient Analysis

Let $r = \log \frac{\pi_\theta(x_f)}{\pi_{\text{ref}}(x_f)}$. Then:

$$\mathcal{L}_{\text{NPO}} = -\frac{2}{\beta} \log \sigma(-\beta r) = \frac{2}{\beta} \log(1 + e^{\beta r})$$

The gradient with respect to $r$ is:

$$\frac{\partial \mathcal{L}_{\text{NPO}}}{\partial r} = \frac{2 e^{\beta r}}{1 + e^{\beta r}} = 2\sigma(\beta r)$$

And the gradient with respect to model parameters $\theta$:

$$\frac{\partial \mathcal{L}_{\text{NPO}}}{\partial \theta} = 2\sigma(\beta r) \cdot \frac{\partial \log \pi_\theta(x_f)}{\partial \theta}$$

**Interpretation**: $\sigma(\beta r)$ acts as an **adaptive weight**:

| $r$ value | $\sigma(\beta r)$ | Meaning |
|-----------|------------------|---------|
| $r \gg 0$ | $\approx 1$ | Model still assigns high probability to $x_f$ → large gradient → strong push to forget |
| $r \approx 0$ | $\approx 0.5$ | Model is near the reference → moderate gradient |
| $r \ll 0$ | $\approx 0$ | Model has already "forgotten" $x_f$ → near-zero gradient → training stabilises |

This self-regularising property prevents **over-forgetting**: the gradient naturally decays as the model forgets more.

### 2.2 The Retain Term (NPO+RT)

NPO alone cannot prevent collateral damage to non-forget capabilities. NPO+RT adds a standard cross-entropy retain loss:

$$\mathcal{L}_{\text{NPO+RT}} = \mathcal{L}_{\text{NPO}} + \gamma \cdot \mathcal{L}_{\text{CE}}(\mathcal{D}_r)$$

where:

$$\mathcal{L}_{\text{CE}}(\mathcal{D}_r) = -\mathbb{E}_{x_r \sim \mathcal{D}_r}\!\left[\frac{1}{|x_r|}\sum_{t=1}^{|x_r|} \log \pi_\theta(x_{r,t} \mid x_{r,<t})\right]$$

The hyperparameter $\gamma$ controls the forget–retain trade-off. We use $\gamma = 1.0$ (paper's recommended value).

---

## 3. Per-Sequence Log-Probability Computation

The key computational challenge in NPO is computing $\log \pi_\theta(x_f)$ efficiently in a batched, numerically stable manner.

### 3.1 Batch computation

For a batch of sequences $\{x_f^{(i)}\}_{i=1}^B$ with variable lengths, we compute:

```python
def get_per_sequence_logprobs(logits, input_ids, attention_mask):
    # logits: [B, T, V], input_ids: [B, T], mask: [B, T]
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # [B, T-1, V]
    # Gather the log-prob of the actual next token
    token_log_probs = log_probs.gather(
        -1, input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)                                          # [B, T-1]
    # Sum over non-padding tokens only
    masked = token_log_probs * attention_mask[:, 1:].float()
    return masked.sum(dim=-1)                              # [B] — per-sequence
```

The output is $\log \pi_\theta(x_f^{(i)}) = \sum_{t=1}^{T_i-1} \log \pi_\theta(x_{f,t+1}^{(i)} \mid x_{f,1:t}^{(i)})$ for each sequence independently.

### 3.2 Reference Model via LoRA disable_adapter()

A key implementation insight: rather than loading two separate model copies (which would require $\sim 5$ GB VRAM), we use **PEFT's `disable_adapter()` context manager**:

```python
with model.disable_adapter():
    model.eval()          # disable OPT's residual dropout (p=0.1)
    with torch.no_grad():
        out_ref = model(input_ids, attention_mask)
    # → this is the REFERENCE model (base OPT-1.3B, no LoRA)
```

This works because the trainable parameters live exclusively in the LoRA adapter layers. Disabling them restores the exact pre-training distribution $\pi_{\text{ref}}$.

**Critical**: `model.eval()` must be called inside `disable_adapter()` because OPT-1.3B has **residual dropout** ($p = 0.1$). Without `eval()`, the reference forward pass is stochastic, introducing noise into the log-ratio $r$.

---

## 4. Training Dynamics: Observed Loss Curves

From `log_npo.txt`, the NPO+RT training loss shows:

**NPO+RT (phase 2), steps 1–500:**

| Step | $\mathcal{L}_{\text{NPO}}$ | $\mathcal{L}_{\text{RT}}$ |
|------|--------------------------|--------------------------|
| 25 | 11.7310 | 3.6020 |
| 100 | 4.3873 | 4.1008 |
| 250 | 1.8696 | 4.0575 |
| 500 | 1.0292 | 3.9609 |

**Key observations**:

1. **$\mathcal{L}_{\text{NPO}}$ decreases monotonically**: from 11.73 to 1.03 — the model is successfully "un-learning" the toxic sequences.

2. **$\mathcal{L}_{\text{RT}}$ stabilises around 4.0**: the retain loss quickly reaches a steady state, meaning the model is not degrading its clean-data generation. This confirms $\gamma=1.0$ provides adequate regularisation.

3. **High initial NPO loss** (11.73) indicates the pretrained model assigns high probability to toxic sequences — confirming Civil Comments toxicity IS encoded in the model weights.

4. **Convergence**: both losses plateau after ~300 steps, suggesting earlier stopping is possible (e.g., 300 steps instead of 500).

### Loss ratio analysis

At step 500: $\mathcal{L}_{\text{NPO}} / (\gamma \cdot \mathcal{L}_{\text{RT}}) = 1.03 / 3.96 \approx 0.26$. The retain term dominates in the final stages, which is desirable — it prevents over-forgetting by anchoring the model to the retain distribution.

---

## 5. The PPL Catastrophe: Why Retain Loss is Essential

From `log_npo.txt`:

```
NPO (forget only): avg_toxicity=0.012, toxic_ratio=0.0, PPL=8,060,184,892,600.66
```

$\text{PPL} \approx 8 \times 10^{12}$ is a catastrophic failure. This is the most important empirical finding in this project.

### Mathematical explanation

The NPO loss without a retain term optimises:

$$\hat{\theta} = \arg\min_\theta \mathcal{L}_{\text{NPO}}(\theta; \mathcal{D}_f)$$

This is equivalent to **minimising** $\log \pi_\theta(x_f)$ subject to no constraint on $\pi_\theta(x_r)$. The unconstrained gradient pushes the model toward regions of weight space where $\pi_\theta(x_f) \to 0$, but these regions are not necessarily close to the surface of any reasonable language model manifold.

In practice, the model learns to produce near-zero probabilities for ALL tokens (not just toxic ones), leading to:

$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i} \log p_\theta(w_i)\right) \approx \exp(28) \approx 10^{12}$$

This confirms that **the forget and retain objectives are inextricably coupled in a language model**: you cannot forget a distribution without simultaneously affecting what the model assigns probability to elsewhere.

---

## 6. Kaggle Implementation Optimizations

### 6.1 LoRA with `TaskType.CAUSAL_LM`

Using PEFT's `TaskType.CAUSAL_LM` ensures the adapter is properly initialised for autoregressive generation. Target modules `['q_proj', 'v_proj']` cover the attention pathways most relevant to content generation (vs `fc1`, `fc2` which are more syntactic).

### 6.2 Linear warmup schedule

```python
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=50, num_training_steps=500
)
```

Warmup prevents large gradient updates in the first 50 steps (10% of training) where the NPO log-ratio is large and potentially unstable.

### 6.3 GradScaler for fp16 stability

The NPO loss involves $\log \sigma(-\beta r)$. When $\beta r \gg 0$ (early training), $-\beta r \ll 0$, and $\sigma(-\beta r) \approx e^{-\beta r} \approx 0$ in fp16 **underflows to zero**, producing NaN gradients. `GradScaler` multiplies the loss by a large factor before backward, preventing this underflow.

### 6.4 Gradient clipping at 1.0

The adaptive weight $2\sigma(\beta r)$ is bounded in $[0, 2]$, but the underlying $\partial \log \pi_\theta / \partial \theta$ can be large early in training. `clip_grad_norm_(params, 1.0)` prevents gradient explosion.
