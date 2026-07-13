---
marp: true
math: mathjax
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 22px;
  }
  h1 {
    font-size: 38px;
    color: #2d3436;
  }
  h2 {
    font-size: 30px;
    color: #6c5ce7;
  }
  h3 {
    font-size: 24px;
    color: #636e72;
  }
  table {
    font-size: 17px;
    margin: 0 auto;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    align-items: center;
  }
  .columns-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 16px;
    align-items: start;
  }
  blockquote {
    font-size: 20px;
    border-left: 5px solid #6c5ce7;
    background: #f8f9fa;
    padding: 10px 20px;
  }
  .caption {
    text-align: center;
    font-size: 13px;
    color: #636e72;
    font-style: italic;
    margin-top: 6px;
  }
  .tag {
    display: inline-block;
    background: #6c5ce7;
    color: white;
    font-size: 14px;
    padding: 2px 10px;
    border-radius: 12px;
    margin-right: 4px;
  }
  .tag-red {
    display: inline-block;
    background: #d63031;
    color: white;
    font-size: 14px;
    padding: 2px 10px;
    border-radius: 12px;
    margin-right: 4px;
  }
  .tag-green {
    display: inline-block;
    background: #00b894;
    color: white;
    font-size: 14px;
    padding: 2px 10px;
    border-radius: 12px;
    margin-right: 4px;
  }
  section.lead h1 {
    font-size: 42px;
    text-align: center;
  }
  section.lead h2 {
    text-align: center;
    color: #6c5ce7;
  }
---

<!-- _class: lead -->

# Octopus
## History-Free Gradient Orthogonalization for Continual Learning in MLLMs

<hr style="border: 1px solid #dfe6e9; margin: 20px 0;">

<div style="text-align: center; font-size: 18px; color: #636e72;">
Yuehao Liu et al. · CVPR 2026
</div>

<div style="text-align: center; margin-top: 40px; font-size: 22px;">2026.07</div>
<div style="text-align: center; margin-top: 10px; font-size: 24px; font-weight: bold;">Hyuntaek Seo</div>

---

## Agenda

1. **Problem** — Catastrophic Forgetting in MLLM Continual Learning
2. **Key Insight** — Taylor Expansion → True Lossless Condition
3. **Method** — GPWC & HiFGO
4. **Method** — Two-stage Finetuning
5. **Experiments** — UCIT Benchmark Setup
6. **Results** — Main Comparison (SOTA)
7. **Ablation** — GPWC vs. Parameter Orthogonality
8. **Ablation** — 2-stage Strategy & λ Sensitivity
9. **Ablation** — Order Robustness
10. **Conclusion**

---

## Problem: Catastrophic Forgetting in MLLM CL

<div class="columns" style="align-items: start;">
<div>

### Setting
- Sequential tasks $T_1, T_2, \dots, T_N$ arrive one by one
- Previous data **inaccessible** after task ends
- Single LoRA adapted sequentially on MLLM

### Prior Work Fails

| Method | Problem |
|:---|:---|
| **Architecture-based** (HiDe-LLaVA) | Inference overhead, generalization drop |
| **Rehearsal-based** | Privacy risk, memory cost |
| **Regularization-based** (O-LoRA) | Param. orthogonality ≠ gradient interference prevention |
| **OGD** | Requires historical data |

</div>
<div>

### Core Challenge

$$\mathcal{L}_{\mathcal{D}_1}(\theta_1^\prime) \ll \mathcal{L}_{\mathcal{D}_1}(\theta_1^\prime + \theta_2)$$

* New task learning **destroys** old task performance.

</div>
</div>

---

## Key Insight: What is the True Lossless Condition?

<div class="columns">
<div>

### Lossless Condition (Eq. 4)
Task 1 performance must not degrade after learning Task 2:

$$\mathcal{L}_{\mathcal{D}_1}(\theta_1^\prime) = \mathcal{L}_{\mathcal{D}_1}(\theta_1^\prime + \theta_2)$$

### Taylor Expansion (Eq. 5)

$$\mathcal{L}_{\mathcal{D}_1}(\theta_1^\prime + \theta_2) \approx \mathcal{L}_{\mathcal{D}_1}(\theta_1^\prime) + \left\langle \frac{\partial \mathcal{L}_{\mathcal{D}_1}(\theta_1^\prime)}{\partial \theta_1^\prime},\; \theta_2 \right\rangle$$

LoRA weights are small → higher-order terms negligible

### Result (Eq. 6)

$$\boxed{\left\langle \frac{\partial \mathcal{L}}{\partial \theta_1^\prime},\; \theta_2 \right\rangle = 0}$$

**Gradient orthogonality** is the true lossless condition

</div>
<div>

### Why NOT Parameter Orthogonality?

**EWC / O-LoRA** 
* Enforce $\langle \theta_1, \theta_2 \rangle = 0$
* $\theta_1$ = *accumulated trajectory*, not gradient direction
→ Does **NOT** prevent gradient interference

### Why NOT Existing Gradient Orthogonality (OGD)?

* Requires old data $\mathcal{D}_1$ to compute gradients
→ Privacy & memory issue

* Converged weights → gradient **weak & oscillatory**

</div>
</div>

---

## Solution: GPWC + HiFGO

<div style="text-align: center;">
  <img src="../images/octopus/figure2.png" style="max-height: 420px; border-radius:8px; box-shadow: 0 6px 20px rgba(0,0,0,0.12);" />
  <div class="caption">Liu et al., "Octopus" · CVPR 2026 · Figure 2: Overview of HiFGO and Two-stage Finetuning</div>
</div>

---

## GPWC: Gradients of Previous parameters Within Current data

<div class="columns">
<div>

### Key Idea
No access to old data $\mathcal{D}_j$ — instead, pass **current data** through **previous weights**:

$$\text{GPWC} = \frac{\partial \mathcal{L}_{\mathcal{D}_i}(\theta_j^\prime)}{\partial \theta_j^\prime}$$

- $\theta_j^\prime = W_0 + \sum_{m=1}^{j} \theta_m$ : accumulated past weights
- Evaluated on $\mathcal{D}_i$ → non-zero, reliable signal

### Why does it work?
+ Converged $\theta_j^\prime$ on **new data** gives a meaningful gradient direction
* Reveals where the update **conflict** with past knowledge

</div>
<div>

### Orthogonality Loss (Eq. 7)

$$\mathcal{L}_{\text{orth}}(\theta_i) = \sum_{j=1}^{i-1} \left( \frac{\partial \mathcal{L}_{\mathcal{D}_i}(\theta_j^\prime)}{\partial \theta_j^\prime} \right)^{\!T} \theta_i$$

Minimizing $\mathcal{L}_{\text{orth}} \to 0$ forces:

$$\theta_i \perp \text{GPWC}_j \quad \forall j < i$$

**No historical data needed.**

<div style="background:#f0f0ff; border-left:5px solid #6c5ce7; padding:12px; border-radius:6px; margin-top:14px;">
<strong>HiFGO</strong> = History-Free Gradient Orthogonalization<br/>
Use current data + past weights to approximate the true gradient constraint
</div>

</div>
</div>

---

## Two-stage Finetuning: Why Needed?

<div class="columns" style="align-items: start;">
<div>

### Problem with Naive Regularization

Applying HiFGO + L2 from the **start**:
- Shrinks effective parameter search space
- $\mathcal{L}_{ce}$ vs $\mathcal{L}_{\text{orth}}$ **compete** → stuck in suboptimal local minima
- Performance collapses (especially after task 3+)

w/o 2-stage: **61.18%**
w/ 2-stage: **71.01%** (+9.83%p)

</div>
<div>

<div style="text-align: center;">
<img src="../images/octopus/figure5b.png" style="width:80%; border-radius:8px; box-shadow:0 4px 14px rgba(0,0,0,0.1);" />
<div class="caption">Liu et al., "Octopus" · CVPR 2026 · Figure 5b: Performance collapse without 2-stage</div>
</div>

</div>
</div>

---

## Two-stage Finetuning: Algorithm

<div class="columns">
<div>

### Stage 1 — Free Adaptation

Disable all regularization. 
Train freely to reach optimal manifold:

$$\mathcal{L}_1 = \frac{1}{|\mathcal{D}_i|} \sum \mathcal{L}_{\text{ce}}(f_{\theta_{i,1}^\prime}(x_k),\, y_k)$$

- Goal 
: Enter the neighborhood of the optimal solution
- No HiFGO, no L2

</div>
<div>

### Stage 2 — Regularized Refinement

Initialize from Stage 1 weights. Activate HiFGO + L2:

$$\mathcal{L}_2 = \frac{1}{|\mathcal{D}_i|} \sum \Big( \mathcal{L}_{\text{ce}} + \lambda_1 \mathcal{L}_{\text{orth}} + \lambda_2 \mathcal{L}_{\text{norm}} \Big)$$

- $\mathcal{L}_{\text{orth}}$: gradient orthogonality (HiFGO)
- $\mathcal{L}_{\text{norm}}$: L2 regularization to keep $\|\theta_i\|$ small (validates Taylor approximation)
- Fine-tune near the optimum **without** competition between objectives

</div>
</div>

---

## Experimental Setup: UCIT Benchmark

<div class="columns">
<div>

### 6 Sequential Domains

| # | Dataset | Domain |
|:---:|:---|:---|
| 1 | ImageNet-R | Image Classification |
| 2 | ArXivQA | Academic QA |
| 3 | VizWiz | Visual QA |
| 4 | IconQA | Icon Recognition QA |
| 5 | CLEVR-Math | Math Reasoning |
| 6 | Flickr30k | Image Captioning |

**Backbone**: LLaVA (MLLM) + Single LoRA

</div>
<div>

### Evaluation Metrics

| Metric | Description |
|:---|:---|
| **Last** | Avg. accuracy after all 6 tasks *(primary)* |
| **Avg** | Avg. accuracy across all stages |
| **Imd.** | Accuracy right after learning each task (upper bound) |
| **BWT** | Backward Transfer — positive = knowledge gain |

### Baselines
- Architecture-based: HiDe-LLaVA, MoILE
- Rehearsal-based: Vanilla Rehearsal
- Regularization-based: EWC, O-LoRA, LwF
- Gradient-based: OGD, InfLoRA

</div>
</div>

---

## Main Results: Radar Chart

<div style="text-align: center;">
  <img src="../images/octopus/figure1.png" style="max-height: 430px; border-radius:8px; box-shadow: 0 6px 20px rgba(0,0,0,0.12);" />
  <div class="caption">Liu et al., "Octopus" · CVPR 2026 · Figure 1: Per-task performance across all methods on UCIT</div>
</div>

---

## Main Results: Quantitative Comparison

<div style="text-align: center;">
  <img src="../images/octopus/table1.png" style="max-height: 380px; border-radius:8px; box-shadow: 0 6px 20px rgba(0,0,0,0.12);" />
  <div class="caption">Liu et al., "Octopus" · CVPR 2026 · Table 1: UCIT Benchmark Results</div>
</div>

| | vs. SOTA (HiDe-LLaVA) | vs. Rehearsal | vs. O-LoRA |
|:---|:---:|:---:|:---:|
| **Octopus** | Avg +2.14%p / Last +6.82%p | Last 71.01% vs 68.44% (no data) | Avg +6.54%p / Last +12.65%p |

---

## Ablation: HiFGO Recovery Capability

<div style="text-align: center;">
  <img src="../images/octopus/figure3.png" style="max-height: 370px; border-radius:8px; box-shadow: 0 6px 20px rgba(0,0,0,0.12);" />
  <div class="caption">Liu et al., "Octopus" · CVPR 2026 · Figure 3: Task 1 performance after learning Task 2</div>
</div>

> **Seq. FT** (no reg.) → Task 1 collapses &nbsp;|&nbsp; **Seq. FT w/ HiFGO** → recovers to **Sgl. FT** level (single-task upper bound)

---

## Ablation: What to Orthogonalize?

<div class="columns">
<div>

<img src="../images/octopus/table2.png" style="width:100%; border-radius:8px; box-shadow:0 4px 14px rgba(0,0,0,0.1);" />
<div class="caption">Liu et al., "Octopus" · CVPR 2026 · Table 2: Orthogonality target comparison</div>

</div>
<div>

<img src="../images/octopus/figure4.png" style="width:100%; border-radius:8px; box-shadow:0 4px 14px rgba(0,0,0,0.1);" />
<div class="caption">Liu et al., "Octopus" · CVPR 2026 · Figure 4: Visualization of parameter space</div>

</div>
</div>

| Orthogonalize target | Last | BWT |
|:---|:---:|:---:|
| Param. vectors ($\theta_1 \perp \theta_2$) | 66.71% | -2.51 |
| **GPWC — current data** (ours) | **71.01%** | **+0.41** |
| Prev. task gradient (old data, OGD-style) | 62.25% | -8.95 |

---

## Ablation: 2-stage & λ Sensitivity

<div class="columns">
<div>

<img src="../images/octopus/table3.png" style="width:100%; border-radius:8px; box-shadow:0 4px 14px rgba(0,0,0,0.1);" />
<div class="caption">Liu et al., "Octopus" · CVPR 2026 · Table 3: Two-stage ablation</div>

<br/>

**w/ 2-stage: 71.01% &nbsp;|&nbsp; w/o: 61.18%**

</div>
<div>

<img src="../images/octopus/table4.png" style="width:100%; border-radius:8px; box-shadow:0 4px 14px rgba(0,0,0,0.1);" />
<div class="caption">Liu et al., "Octopus" · CVPR 2026 · Table 4: λ (L2 strength) ablation</div>

<br/>

- Too small → high forgetting (BWT -8.59)
- Too large → low plasticity (Imd. 64.12%)
- **Optimal: $\lambda = 1\mathrm{e}{-2}$**

</div>
</div>

---

## Ablation: Order Robustness

<div class="columns">
<div>

<img src="../images/octopus/figure5a.png" style="width:100%; border-radius:8px; box-shadow:0 4px 14px rgba(0,0,0,0.1);" />
<div class="caption">Liu et al., "Octopus" · CVPR 2026 · Figure 5a: Performance curves under 3 task orders</div>

</div>
<div>

<img src="../images/octopus/table5.png" style="width:100%; border-radius:8px; box-shadow:0 4px 14px rgba(0,0,0,0.1);" />
<div class="caption">Liu et al., "Octopus" · CVPR 2026 · Table 5: Order robustness results</div>

</div>
</div>

<br/>

| Task Order | Last |
|:---:|:---:|
| R-A-V-I-C-F | 71.01% |
| A-I-R-F-C-V | 69.04% |
| I-F-R-C-A-V | 69.16% |

> Performance curves nearly **identical** across all orderings — **Order-invariant Robustness**

---

## Conclusion

<div class="columns">
<div>

### Contributions

1. **Theoretically proved** (Taylor expansion) that gradient orthogonality is the true lossless condition for CL

2. **GPWC** — computes meaningful gradient signals from previous weights using *current data*, removing dependence on historical data

3. **2-stage finetuning** — separates free learning from constrained refinement, eliminating regularization-plasticity competition

</div>
<div>

### Pros & Cons

### Pros
- No historical data (privacy & memory free)
- Single LoRA — no extra modules
- Order-invariant robustness

### Cons
- GPWC cost grows linearly with task count
- λ hyperparameter sensitivity
- Limited CL scenario coverage

</div>
</div>

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Thank You
## Questions?
