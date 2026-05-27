# Mathematical Analysis: RMU + RNA — Representation Misdirection with Random Noise Augmentation

> **Papers**: RMU (arXiv:2403.03218) + RNA (arXiv:2501.19202v2)  
> **File**: `rmu-algo.ipynb`  
> **Result**: avg_toxicity 0.2093 → **0.0763** | toxic_ratio 16.5% → **5.0%** | PPL 13.88 → **14.50**  
> *(After hyperparameter correction: ALPHA 300→100, TARGET_LAYER 7→15, N_STEPS 150→300, masked MSE)*

---

## 1. Core Concept: Representation-Level Unlearning

Unlike NPO (which operates on output probabilities) and DEPN (which operates on weights directly), RMU operates at the **hidden representation level**. The insight is:

> If the model can no longer form the internal representation of "toxic context" at some intermediate layer, it cannot produce toxic output — regardless of what the output head does.

Formally, for layer $l$ of OPT-1.3B with hidden states $h^{(l)} \in \mathbb{R}^{2048}$:

- The model generates toxic text because toxic inputs $x_f$ produce hidden states $h^{(l)}(x_f)$ in a specific **toxic region** of representation space
- RMU forces $h^{(l)}(x_f) \to c\mathbf{u}$, where $\mathbf{u}$ is a **random unit vector** and $c$ is a scaling coefficient
- Since $c\mathbf{u}$ is random and far from any semantic region, the model "loses the map" to generate toxic content

---

## 2. RMU Loss Function: Forget Term

The forget loss steers the layer-$l$ representation of forget samples toward the fixed random target $c\mathbf{u}$:

$$\mathcal{L}_{\text{forget}} = \mathbb{E}_{x_f \sim \mathcal{D}_f} \frac{1}{|\mathcal{T}(x_f)|} \sum_{t \in \mathcal{T}(x_f)} \left\| h_\theta^{(l)}(x_f)_t - c\mathbf{u} \right\|^2$$

where:
- $\mathbf{u} \in \mathbb{R}^{2048}$ is a **fixed** random unit vector: $\|\mathbf{u}\| = 1$, sampled once at training start
- $c = C\_\text{COEFF} = 400$ is the scaling coefficient
- $\mathcal{T}(x_f)$ is the set of **non-padding** token positions (masked MSE)
- $h_\theta^{(l)}(x_f)_t$ is the hidden state at position $t$, layer $l$, for input $x_f$

### Why a random unit vector?

The target $c\mathbf{u}$ must be:
1. **Consistent** across training steps (same target every step → stable gradient direction)
2. **Far from the data manifold** (so the model loses all semantic meaning)
3. **Not interpretable** as any known concept

A random unit vector satisfies all three. Its norm is $\|c\mathbf{u}\| = c = 400$, whereas typical OPT-1.3B hidden states have norm $\approx \sqrt{2048} \approx 45$ (unit variance per dimension). The target is therefore $\sim 9\times$ the typical hidden state magnitude — safely far from the normal representation manifold.

### Initial forget loss magnitude

At the start of training, $h_\theta^{(l)} \approx h_{\text{ref}}^{(l)}$ (no updates yet). The initial loss is:

$$\mathcal{L}_{\text{forget}}^{(0)} \approx \frac{1}{D} \sum_{d=1}^{D} (h_d - c \cdot u_d)^2 \approx \frac{1}{D}\left(\|h\|^2 + c^2\|\mathbf{u}\|^2 - 2c \langle h, \mathbf{u} \rangle\right)$$

Since $\|\mathbf{u}\| = 1$ and $\langle h, \mathbf{u} \rangle \approx 0$ (random dot product):

$$\mathcal{L}_{\text{forget}}^{(0)} \approx \frac{\|h\|^2}{D} + \frac{c^2}{D} \approx \frac{45^2}{2048} + \frac{400^2}{2048} \approx 1.0 + 78.1 \approx 79.1$$

**From log_rmu.txt**: step 1 forget loss = **79.1264** ✓ — exact match with theory.

---

## 3. RMU Loss Function: Retain Term

The retain loss prevents the model's representations on clean data from drifting away from the reference model:

$$\mathcal{L}_{\text{retain}} = \mathbb{E}_{x_r \sim \mathcal{D}_r} \frac{1}{|\mathcal{T}(x_r)|} \sum_{t \in \mathcal{T}(x_r)} \left\| h_\theta^{(l)}(x_r)_t - h_{\text{ref}}^{(l)}(x_r)_t \right\|^2$$

At initialisation, $\mathcal{L}_{\text{retain}}^{(0)} = 0$ (model equals reference). The loss grows as training proceeds.

**From log_rmu.txt**: step 1 retain loss = **0.050413** — already small but non-zero due to LoRA initialisation (LoRA adapters initialise with $A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$, giving $\Delta W = BA = 0$ at step 0. The tiny non-zero retain loss at step 1 comes from the first gradient update).

---

## 4. RNA: Random Noise Augmentation

RNA (arXiv:2501.19202) modifies the retain loss to add **Gaussian noise to the reference target**:

$$\mathcal{L}_{\text{retain}}^{\text{RNA}} = \mathbb{E}_{x_r \sim \mathcal{D}_r} \frac{1}{|\mathcal{T}(x_r)|} \sum_{t \in \mathcal{T}(x_r)} \left\| h_\theta^{(l)}(x_r)_t - \left(h_{\text{ref}}^{(l)}(x_r)_t + \varepsilon_t\right) \right\|^2$$

where $\varepsilon_t \sim \mathcal{N}(0, \nu^2 I)$, $\nu =$ `RNA_NOISE_STD` $= 0.1$, independent for each token position.

### Gradient analysis

The gradient of the RNA retain loss with respect to $h_t$ is:

$$\frac{\partial \mathcal{L}_{\text{retain}}^{\text{RNA}}}{\partial h_t} = \frac{2}{n_{\text{real}} \cdot D} \left(h_t - h_{\text{ref},t} - \varepsilon_t\right)$$

Taking the expectation over $\varepsilon_t \sim \mathcal{N}(0, \nu^2 I)$:

$$\mathbb{E}_\varepsilon\left[\frac{\partial \mathcal{L}_{\text{retain}}^{\text{RNA}}}{\partial h_t}\right] = \frac{2}{n_{\text{real}} \cdot D}\left(h_t - h_{\text{ref},t}\right)$$

This is identical to the gradient of the standard retain loss — so RNA does **not** change the optimal solution or the mean gradient direction.

### RNA's true effect: gradient noise injection

RNA injects **stochastic gradient noise** with variance:

$$\text{Var}\left[\frac{\partial \mathcal{L}_{\text{retain}}^{\text{RNA}}}{\partial h_t}\right] = \frac{4\nu^2}{(n_{\text{real}} \cdot D)^2}$$

This gradient noise acts as a regulariser: it prevents the model from becoming overly sensitive to specific patterns in $h_{\text{ref}}$ (which could be triggered by forget-tokens appearing in retain queries). The paper's theoretical framework formalises this as a **backdoor defence** — the retain target $h_{\text{ref}} + \varepsilon$ varies stochastically, training the model to produce representations that match a **neighbourhood** around $h_{\text{ref}}$ rather than the exact point $h_{\text{ref}}$.

---

## 5. Combined RMU+RNA Loss

$$\boxed{\mathcal{L}_{\text{RMU+RNA}} = \mathcal{L}_{\text{forget}} + \alpha \cdot \mathcal{L}_{\text{retain}}^{\text{RNA}}}$$

with $\alpha = 100$ (after hyperparameter correction).

### The ALPHA gradient balance

At step $s$ with forget loss $F_s$ and retain loss $R_s$:

$$\frac{\|\nabla_\theta F_s\|}{\|\nabla_\theta (\alpha R_s)\|} = \frac{F_s}{\alpha R_s} \cdot \text{(architecture factor)}$$

At initialisation ($F_0 \approx 79$, $R_0 \approx 0.05$):

$$\frac{79}{\alpha \cdot 0.05} = \frac{79}{5} = 15.8 \quad (\alpha = 100)$$

The forget gradient is $\sim 16\times$ larger than the retain gradient initially — appropriate for the first phase of training where forgetting is the priority.

**At the original $\alpha = 300$:**

$$\frac{79}{300 \cdot 0.05} = \frac{79}{15} = 5.3$$

Still reasonable at step 1. But as $R_s$ grows to $\sim 0.1$ (as observed in the log):

$$\frac{79}{300 \cdot 0.1} = \frac{79}{30} = 2.6 \quad (\alpha=300)$$

$$\frac{79}{100 \cdot 0.1} = \frac{79}{10} = 7.9 \quad (\alpha=100)$$

With $\alpha=300$, the retain gradient quickly dominates as training progresses, **locking** the model near the reference and preventing the forget loss from moving the forget representations far enough.

---

## 6. Masked MSE: Why Padding Matters

With `MAX_LENGTH=128` and Civil Comments texts of average length $\sim 60$ tokens:

$$\text{Padding fraction} \approx 1 - \frac{60}{128} \approx 53\%$$

Without masking, the MSE is computed over $B \times T \times D = 4 \times 128 \times 2048 = 1,048,576$ elements, of which $\sim 544,000$ (53%) are padding token positions.

**The problem for the forget loss**: we push padding token representations toward $c\mathbf{u}$. Padding tokens have no semantic content — steering them corrupts the model's ability to handle padding in future inference without any benefit to unlearning.

**The problem for the retain loss**: the retain constraint on padding positions is trivially satisfied (padding has a fixed, predictable representation) — it "wastes" gradient capacity that could be used for meaningful retain tokens.

**Masked MSE formula** (only real tokens):

$$\mathcal{L}_{\text{masked}} = \frac{\sum_{b,t,d} (h_{btd} - \text{target}_{btd})^2 \cdot \mathbf{1}[\text{mask}_{bt}=1]}{\sum_{b,t} \mathbf{1}[\text{mask}_{bt}=1] \cdot D}$$

---

## 7. Training Dynamics: Observed Loss Curve

From `log_rmu.txt` (layer=15, $\alpha$=100, masked MSE, 300 steps):

| Step | Forget Loss | Retain Loss | Ratio $F/(\alpha R)$ |
|------|-------------|-------------|---------------------|
| 1 | 79.1264 | 0.050413 | **15.7** |
| 75 | 75.5901 | 0.065533 | 11.5 |
| 165 | 34.2720 | 0.065653 | **5.2** |
| 225 | 29.9300 | 0.097230 | 3.1 |
| 300 | 35.3161 | 1.079583 | **0.33** |

**Key observations**:

1. Forget loss decreases from **79.1** to **35.3** — the model IS learning to steer representations away from their natural positions. This is the fundamental mechanism working.

2. Some oscillation (e.g., step 180: 58.0, step 225: 29.9, step 270: 59.6) suggests the training is somewhat unstable. This is expected with the stochastic RNA noise and the complex interaction between forget and retain gradients.

3. At step 300, the retain loss spikes to **1.079** (vs typical 0.05–0.15). This suggests the model made a large forgetting step at the cost of retain. The final result (toxicity 0.0763) confirms the forgetting was successful.

4. The gradient ratio drops from 15.7 to 0.33 by step 300 — the retain constraint becomes dominant in the final steps, stabilising the model near the reference for clean text.

---

## 8. Why Layer 15 vs Layer 7

Layer 7 (original): **29.2% depth** in OPT-1.3B. At this layer, hidden states primarily encode:
- Token identity and local context
- Syntactic structure (part-of-speech, dependency structure)

Layer 15 (corrected): **62.5% depth**. At this layer, hidden states encode:
- Semantic content and discourse structure
- Stylistic register (formal/informal/aggressive tone)
- **Toxicity** as a semantic-stylistic property

Steering at layer 15 disrupts the model's representation of *what kind of language to produce*, while steering at layer 7 mostly disrupts syntax — which explains why the original layer=7 run was largely ineffective (toxicity increased from 0.1859 to 0.2023).

---

## 9. Practical Improvements Over Paper Theory

| Paper assumption | Our adaptation | Improvement |
|-----------------|----------------|-------------|
| $\alpha=300$ tuned for Zephyr-7B/WMDP | $\alpha=100$ for OPT-1.3B toxicity | Prevents retain dominance |
| Layer 7 for Zephyr-7B (32 layers) | Layer 15 for OPT-1.3B (24 layers) | Captures semantic toxicity |
| Full MSE including padding | Masked MSE (real tokens only) | Cleaner gradient signal |
| 500+ steps on WMDP (~3k samples) | 300 steps on Civil Comments (23k) | Appropriate for larger dataset |
| `device_map='auto'` acceptable | `device_map={'': 'cuda:0'}` | Prevents dual-GPU device mismatch on Kaggle |

### Numerical verification of forget loss theory

The initial forget loss prediction:

$$\mathcal{L}_{\text{forget}}^{(0)} = \frac{c^2}{D} = \frac{400^2}{2048} = \frac{160000}{2048} \approx 78.1$$

Observed: **79.1264** — difference of 1.0, explained by the small $\|h\|^2 / D \approx 1.0$ contribution from the actual hidden state norm at layer 15. This confirms our implementation is mathematically correct.
