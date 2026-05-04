# VulMorph-Fed: Cross-Project Software Vulnerability Detection via Federated Vulnerability Morphology Learning
## Research Plan — Targeting: Information and Software Technology (IST), Elsevier

---

## 1. Problem Statement

### 1.1 Motivation

Software vulnerabilities are a critical and growing threat to modern systems. The NVD reported over 25,000 new CVEs in 2024 alone, and DARPA's Artificial Intelligence Cyber Challenge (AIxCC) in 2024 underscored the urgency of automating vulnerability detection. Despite rapid advances in deep learning–based vulnerability detection (DLVD), two fundamental barriers remain unsolved:

**Barrier 1 — The Generalisation Gap.**
State-of-the-art DLVD models fail catastrophically when applied across software projects. A model trained on FFmpeg cannot reliably detect vulnerabilities in OpenSSL, even when both projects contain the same CWE type. Independent replication studies confirm performance drops of 50–90% when moving from within-project to cross-project evaluation settings. The reason is that current models learn project-specific lexical patterns (API names, variable naming conventions, library calls) rather than the underlying structural vulnerability pattern.

**Barrier 2 — The Data Silo Problem.**
Real-world vulnerability data is routinely proprietary. Financial institutions, cloud vendors, and defence contractors cannot share raw source code with external parties due to IP concerns, regulatory constraints, and competitive sensitivity. This means a centralised model trained on one organisation's codebase cannot legally or practically access another's data — even when both organisations would benefit from collaboration. Federated learning (FL) dissolves the data-sharing barrier, but naively applying FL to vulnerability detection (e.g., FedAvg on a GNN backbone) does not solve the generalisation gap, because weight averaging does not transfer structural vulnerability knowledge across heterogeneous codebases.

### 1.2 Root Cause

The root cause of both barriers is the same: **existing models represent code at the lexical/token level, not at the structural/semantic level**. A buffer overflow in FFmpeg and a buffer overflow in OpenSSL are structurally identical in the code property graph (CPG) — both exhibit a loop incrementing an index, an array access at that index, and a missing bounds check — but they use completely different function names and API calls. No existing method exploits this structural invariance for cross-project transfer.

### 1.3 Problem Formulation

Let $\mathcal{C} = \{C_1, C_2, \ldots, C_K\}$ be a set of $K$ software organisations (clients). Each client $C_k$ holds a private dataset $\mathcal{D}_k = \{(G_i^k, y_i^k)\}_{i=1}^{n_k}$ where $G_i^k$ is the code property graph of a function extracted from client $k$'s codebase, and $y_i^k \in \{0, 1\}$ is its vulnerability label. Raw code and datasets are never shared.

We aim to learn a global vulnerability detection model $f_\theta : G \rightarrow [0, 1]$ such that:

1. **Privacy**: No raw code, source tokens, or raw model gradients are transmitted between clients or to the server.
2. **Cross-project generalisation**: $f_\theta$ achieves strong detection performance on held-out projects not seen during training (cross-project split evaluation).
3. **Structural transfer**: $f_\theta$ captures CWE-level vulnerability patterns that are invariant to project-specific lexical variation.
4. **Heterogeneity robustness**: $f_\theta$ handles non-IID distributions across clients (different CWE type distributions, coding styles, programming languages).

### 1.4 Why Existing Approaches Fail

| Approach | Generalisation Gap | Data Silo | Structural Transfer |
|---|---|---|---|
| Devign, IVDetect, ReVeal (centralised GNN) | ✗ fails cross-project | ✗ requires raw code | ✗ lexical features |
| CPVD, VulGDA, CSVD-TF (domain adaptation) | partial | ✗ requires raw code | ✗ lexical features |
| VulFL, Zhang IST 2024 (FL + sequence/token model) | ✗ | ✓ | ✗ lexical features |
| FedAvg + GNN (naive combination) | ✗ | ✓ | ✗ no morphology |
| **VulMorph-Fed (proposed)** | **✓** | **✓** | **✓** |

---

## 2. Literature Review

### 2.1 Deep Learning for Software Vulnerability Detection

**Seminal sequence-based methods**

- Li, Z., Zou, D., Xu, S., Jin, H., Zhu, Y., & Chen, Z. (2018). **VulDeePecker: A Deep Learning-Based System for Vulnerability Detection**. *Network and Distributed System Security Symposium (NDSS) 2018*. [https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-2_Li_paper.pdf]
  > Introduced "code gadgets" (data-flow-based program slices) fed to a BiLSTM; the first function-granularity DL vulnerability detector; released the Code Gadget Database (CGD).

- Li, Z., Zou, D., Xu, S., et al. (2022). **SySeVR: A Framework for Using Deep Learning to Detect Software Vulnerabilities**. *IEEE Transactions on Dependable and Secure Computing (TDSC), 19(4)*. [https://ieeexplore.ieee.org/document/9321538]
  > Systematic syntax/semantics framework producing Syntax-based Vulnerability Candidates (SyVCs) and Semantics-based Vulnerability Candidates (SeVCs); BiLSTM/BiGRU evaluation on NVD+SARD; released the widely-used SeVC dataset.

- Wu, Y., Zou, D., Dou, S., et al. (2022). **VulCNN: An Image-Inspired Scalable Vulnerability Detection System**. *IEEE/ACM International Conference on Software Engineering (ICSE 2022)*. [https://dl.acm.org/doi/10.1145/3510003.3510229]
  > Converts PDGs to images via centrality-encoded sentence embeddings and applies CNNs; approximately 6× faster than Devign; discovered 73 previously unreported real-world vulnerabilities.

**GNN-based methods — foundational**

- Yamaguchi, F., Golde, N., Arp, D., & Rieck, K. (2014). **Modeling and Discovering Vulnerabilities with Code Property Graphs**. *IEEE Symposium on Security and Privacy (S&P) 2014*. [https://ieeexplore.ieee.org/document/6956589]
  > Defined the Code Property Graph (CPG) unifying AST, CFG and PDG; foundation of the Joern static analysis toolkit and all subsequent GNN-based vulnerability detectors.

- Zhou, Y., Liu, S., Siow, J., Du, X., & Liu, Y. (2019). **Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks**. *Advances in Neural Information Processing Systems (NeurIPS 2019)*. [https://arxiv.org/abs/1909.03496]
  > GGNN with Conv pooling on a composite graph (AST+CFG+DFG+NCS) over FFmpeg, QEMU, Wireshark and Linux; the canonical GNN baseline; released the Devign dataset.

- Cheng, X., Wang, H., Hua, J., Xu, G., & Sui, Y. (2021). **DeepWukong: Statically Detecting Software Vulnerabilities Using Deep Graph Neural Networks**. *ACM Transactions on Software Engineering and Methodology (TOSEM), 30(3)*. [https://dl.acm.org/doi/10.1145/3436877]
  > GNN over inter-procedural XFG (program slice + control/data flow); slice-level detection of MITRE Top-10 CWEs on 105K programs.

- Li, Z., Zou, D., Tang, J., et al. (2021). **Vulnerability Detection with Fine-Grained Interpretations**. *ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE 2021)*. [https://dl.acm.org/doi/10.1145/3468264.3468597]
  > IVDetect: Feature-Attention GCN (FA-GCN) over PDG with GNNExplainer-based interpretability; statement-level localisation and explanations.

- Chakraborty, S., Krishna, R., Ding, Y., & Ray, B. (2022). **Deep Learning based Vulnerability Detection: Are We There Yet?** *IEEE Transactions on Software Engineering (TSE), 48(9)*. [https://ieeexplore.ieee.org/document/9448435]
  > Landmark replication study showing >50% performance drop for SOTA DLVDs on realistic cross-project data; releases the ReVeal dataset; proposes GGNN+SMOTE+triplet-loss mitigation.

**GNN-based methods — 2024–2026**

- Sun, X., et al. (2024). **Vul-LMGNNs: Fusing Language Models and Online-Distilled Graph Neural Networks for Code Vulnerability Detection**. *arXiv preprint 2404.14719*. [https://arxiv.org/abs/2404.14719]
  > Integrates CodeBERT node embeddings into a gated GNN over CPGs with an online knowledge-distillation mechanism; joint training surpasses either LM or GNN alone on BigVul and Devign.

- Toan, D., et al. (2025). **Detecting Code Vulnerabilities with Heterogeneous GNN Training (HAGNN)**. *arXiv preprint 2502.16835*. [https://arxiv.org/abs/2502.16835]
  > Inter-Procedural Abstract Graphs (IPAGs) as language-agnostic representation; Heterogeneous Attention GNN achieves 96.6% accuracy on 108-CWE C dataset and 97.8% on 114-CWE Java dataset.

- Ufuktepe, E., et al. (2026). **VulGNN: Software Vulnerability Detection Using a Lightweight Graph Neural Network**. *arXiv preprint 2603.29216*. [https://arxiv.org/abs/2603.29216]
  > Compact GNN matching LLM-based detectors on real-world benchmarks while being 100× smaller; evaluates cross-dataset generalisation; proposes edge deployment in CI/CD pipelines.

**Transformer / code-LLM detectors**

- Fu, M., & Tantithamthavorn, C. (2022). **LineVul: A Transformer-based Line-Level Vulnerability Prediction**. *IEEE/ACM Mining Software Repositories (MSR 2022)*. [https://dl.acm.org/doi/10.1145/3524842.3527927]
  > CodeBERT-based line-level Transformer detector; reports 91% F1 on BigVul (inflated due to dataset duplication, as shown in subsequent replication studies).

- Hin, D., Kan, A., Chen, H., & Babar, M. A. (2022). **LineVD: Statement-level Vulnerability Detection using Graph Neural Networks**. *IEEE/ACM Mining Software Repositories (MSR 2022)*. [https://arxiv.org/abs/2203.05181]
  > Statement-level GNN combining CPG structure with CodeBERT node features; +105% statement-level F1 over prior baselines; demonstrates graph+LM hybrids outperform either alone.

- Liu, Z., et al. (2026). **Efficient Code Analysis via Graph-Guided Large Language Models**. *arXiv preprint 2601.12890*. [https://arxiv.org/abs/2601.12890]
  > Parses project into code graph; LLM encodes nodes; GNN under sparse supervision performs initial detection; GNN prediction backtracking guides LLM attention to suspicious code regions.

- Lekssays, A., et al. (2025). **LLMxCPG: Context-Aware Vulnerability Detection Through Code Property Graph-Guided LLMs**. *arXiv preprint 2507.16585*. [https://arxiv.org/abs/2507.16585]
  > CPG-guided LLM detector achieving 15–40% F1 improvement over SOTA baselines; robust to syntactic code modifications.

**Agentic detectors (ICSE 2026)**

- Weissberg, F., Pirch, L., Imgrund, E., et al. (2026). **LLM-based Vulnerability Discovery through the Lens of Code Metrics**. *IEEE/ACM International Conference on Software Engineering (ICSE 2026)*. [https://www.mlsec.org/docs/2026-icse.pdf]
  > Critical finding: a classifier trained solely on 23 syntactic code metrics performs on par with SOTA LLMs for vulnerability discovery; LLMs operate at a similarly shallow structural level. Directly motivates VulMorph-Fed's deep structural approach.

- Widyasari, R., Weyssow, M., Irsan, I. C., et al. (2026). **VulTrial: A Mock-Court Approach to Vulnerability Detection Using LLM-Based Agents**. *IEEE/ACM International Conference on Software Engineering (ICSE 2026)*. [https://arxiv.org/abs/2505.10961]
  > Courtroom-inspired multi-agent framework (prosecutor/defense/judge/jury agents); achieves 18.6% on PrimeVul Paired with GPT-4o; shows multi-perspective reasoning helps but remains far below production-grade accuracy.

**Replication and generalisation studies**

- Chen, Y., Ding, Z., Alowain, L., Chen, X., & Wagner, D. (2023). **DiverseVul: A New Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection**. *ACM Conference on Computer and Communications Security — RAID 2023*. [https://arxiv.org/abs/2304.00409]
  > 18,945 vulnerable + 330,492 non-vulnerable C/C++ functions from 295+ new projects; shows increasing training data alone does not solve generalisation.

- Ding, Y., Fu, Y., Ibrahim, O., et al. (2025). **Vulnerability Detection with Code Language Models: How Far Are We? (PrimeVul)**. *IEEE/ACM International Conference on Software Engineering (ICSE 2025)*. [https://arxiv.org/abs/2403.18624]
  > PrimeVul dataset with rigorous paired vuln/patch test and temporal split; GPT-4 achieves <12% accuracy; F1 drops from 68% (BigVul) to 3% (PrimeVul) for a 7B code LM.

- Daoudi, N., et al. (2024). **Revisiting the Performance of Deep Learning-Based Vulnerability Detection on Realistic Datasets (Real-Vul)**. *IEEE Transactions on Software Engineering (TSE 2024)*. [https://arxiv.org/abs/2407.03093]
  > Evaluates DeepWukong, LineVul, ReVeal and IVDetect on repository-scale scanning; all models degrade sharply on entire codebases with realistic class imbalance.

- Shereen, E., et al. (2024). **SoK: On Closing the Applicability Gap in Automated Vulnerability Detection**. *arXiv preprint 2412.11194*. [https://arxiv.org/abs/2412.11194]
  > Systematises 79 AVD articles; 89.9% focus on binary classification of C/C++ at function level; identifies declining novel vulnerability discovery and narrow language coverage.

- Shimmi, S., Okhravi, H., & Rahimi, M. (2025). **AI-Based Software Vulnerability Detection: A Systematic Literature Review**. *arXiv preprint 2506.10280*. [https://arxiv.org/abs/2506.10280]
  > SLR covering 2018–2023; graph-based models are most prevalent (91% of AI-based studies); highlights federated learning as explicitly underexplored.

### 2.2 Cross-Project Vulnerability and Defect Prediction

- Liu, S., et al. (2022). **CD-VulD: Cross-Domain Vulnerability Discovery Based on Deep Domain Adaptation**. *IEEE Transactions on Dependable and Secure Computing (TDSC 2022)*. [https://ieeexplore.ieee.org/document/9054952]
  > First framework targeting cross-domain vulnerability discovery via metric transfer learning; requires centralised raw code access from both source and target domains.

- Zhang, X., et al. (2023). **CPVD: Cross Project Vulnerability Detection Based on Graph Attention Network and Domain Adaptation**. *IEEE Transactions on Software Engineering (TSE 2023)*. [https://dl.acm.org/doi/abs/10.1109/TSE.2023.3285910]
  > GAT + Convolution Pooling on CPGs combined with adversarial domain adaptation; current SOTA for fully supervised cross-project SVD; requires centralised code sharing.

- Tian, Y., et al. (2024). **CSVD-TF: Cross-Project Software Vulnerability Detection with TrAdaBoost by Fusing Expert and Semantic Metrics**. *Information and Software Technology (IST), Elsevier 2024*. [https://www.sciencedirect.com/science/article/abs/pii/S0164121224000815]
  > TrAdaBoost-based ensemble fusing CK/Halstead and code2vec metrics; published in IST — confirms the target journal's receptivity to cross-project SVD.

- Li, Z., et al. (2023). **Cross-Domain Vulnerability Detection Using Graph Embedding and Domain Adaptation (VulGDA)**. *Computers & Security, 2023*. [https://www.sciencedirect.com/science/article/abs/pii/S0167404822004096]
  > Zero-shot cross-domain transfer via deep domain adaptation on graph embeddings; requires centralised code access.

- Yamamoto, H., Wang, D., Rajbahadur, G. K., et al. (2023). **Towards Privacy-Preserving Cross-Project Defect Prediction with Federated Learning (FLR)**. *IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER 2023)*. [https://ieeexplore.ieee.org/document/10123655]
  > First FL-based CPDP method; FedAvg on 25 projects using software metrics; no code graphs.

- Wang, Y., et al. (2025). **FedDP: Better Knowledge Enhancement for Privacy-Preserving Cross-Project Defect Prediction**. *Journal of Software: Evolution and Process 2025*. [https://dl.acm.org/doi/abs/10.1002/smr.2761]
  > Local heterogeneity awareness + global knowledge distillation for FL-based CPDP; 19-project evaluation; tabular metrics only.

### 2.3 Federated Learning for Software Engineering

- Shanbhag, A., & Chimalakonda, S. (2022). **Exploring the Under-Explored Terrain of Non-Open-Source Data for Software Engineering Through the Lens of Federated Learning**. *ACM Joint European Software Engineering Conference (ESEC/FSE NIER 2022)*. [https://dl.acm.org/doi/10.1145/3540250.3560879]
  > Vision paper first explicitly motivating FL for SE with proprietary source code.

- Yang, Y., Hu, X., Gao, Z., et al. (2024). **Federated Learning for Software Engineering: A Case Study of Code Clone Detection and Defect Prediction**. *IEEE Transactions on Software Engineering (TSE 50(2), 2024)*. [https://ieeexplore.ieee.org/document/10310027]
  > Most comprehensive FL-SE empirical study; CodeBERT and metric-vector clients; no GNN or code-graph client evaluated. Confirms non-IID is the dominant challenge.

- Zhang, X., Yu, Z., Liu, X., & Xin, Y. (2024). **Vulnerability Detection Based on Federated Learning**. *Information and Software Technology (IST), Elsevier 2024*. [https://dl.acm.org/doi/10.1016/j.infsof.2023.107371]
  > First IST paper applying FL to vulnerability detection using sequence/token model; competitive with centralised but no graphs or cross-project heterogeneity treatment.

- Hu, M., Quan, X., Peng, Y., et al. (2024). **VulFL: An Empirical Study of Vulnerability Detection using Federated Learning**. *arXiv preprint 2411.16099*. [https://arxiv.org/abs/2411.16099]
  > VulFL framework tests FL with NLP (CodeBERT, CodeLlama) and GNN (Devign-style) backbones across CWE partitions of DiverseVul; explicitly notes FL-GNN over CPGs warrants deeper investigation.

- Cai, F., et al. (2025). **FedMVA: Enhancing Software Vulnerability Assessment via Federated Multimodal Learning**. *Journal of Systems and Software 2025*. [https://www.sciencedirect.com/science/article/abs/pii/S0164121225001372]
  > Federated multimodal vulnerability assessment; code structure + lexical features + developer comments; token-level features only.

- Drvshavva, O., et al. (2025). **Privacy-Preserving Methods for Bug Severity Prediction**. *Evaluation and Assessment in Software Engineering (EASE 2025)*. [https://arxiv.org/abs/2506.22752]
  > FL + synthetic data (GAN) for bug severity prediction; FL-trained models comparable to centralised without data sharing.

- He, C., Balasubramanian, K., et al. (2021). **FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks**. *ICLR Workshop on Distributed and Private Machine Learning (DPML 2021)*. [https://arxiv.org/abs/2104.07145]
  > First federated GNN benchmark on molecules/citation networks; demonstrates federated GNNs underperform centralised on non-IID graph splits; motivates the proposed work.

- Wang, L., Liu, J., Gao, X., et al. (2026). **FedDense: Tackling Non-IID Graphs via Decoupled Structure and Feature in Federated Graph Learning**. *Database Systems for Advanced Applications (DASFAA 2026), Springer LNCS 15988*. [https://link.springer.com/chapter/10.1007/978-981-95-3906-2_22]
  > Decouples structural and feature learning in federated GNN; structural GNN channel shared globally while feature learning stays local; methodological template for CPG federation.

### 2.4 Privacy-Preserving Mechanisms

- Abadi, M., Chu, A., Goodfellow, I., et al. (2016). **Deep Learning with Differential Privacy (DP-SGD)**. *ACM Conference on Computer and Communications Security (CCS 2016)*. [https://dl.acm.org/doi/10.1145/2976749.2978318]
  > Per-sample gradient clipping + Gaussian noise; standard DP training algorithm for FL clients.

- Bonawitz, K., Ivanov, V., Kreuter, B., et al. (2017). **Practical Secure Aggregation for Privacy-Preserving Machine Learning**. *ACM Conference on Computer and Communications Security (CCS 2017)*. [https://dl.acm.org/doi/10.1145/3133956.3133982]
  > Cryptographic secure aggregation enabling FL without server seeing individual client updates.

- McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). **Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)**. *International Conference on Artificial Intelligence and Statistics (AISTATS 2017)*. [https://arxiv.org/abs/1602.05629]

- Li, T., Sahu, A. K., Zaheer, M., et al. (2020). **Federated Optimization in Heterogeneous Networks (FedProx)**. *Proceedings of Machine Learning and Systems (MLSys 2020)*. [https://arxiv.org/abs/1812.06127]

- Karimireddy, S. P., Kale, S., Mohri, M., et al. (2020). **SCAFFOLD: Stochastic Controlled Averaging for Federated Learning**. *International Conference on Machine Learning (ICML 2020)*. [https://arxiv.org/abs/1910.06378]

---

## 3. Proposed Algorithm: VulMorph-Fed

### 3.1 Overview

**VulMorph-Fed** (*Vulnerability Morphology-Guided Federated Learning*) is a novel three-phase federated learning framework for cross-project vulnerability detection. The core insight is:

> **Vulnerability patterns are structurally invariant across projects but lexically heterogeneous.** A buffer overflow in FFmpeg and one in OpenSSL share identical control/data-flow topology but use different APIs and variable names. No existing method exploits this structural invariance for cross-project transfer.

The algorithm introduces three chained innovations, each addressing a distinct failure mode of existing approaches:

```
Phase 1 (Client):  Raw CPG  →  VCSA  →  Vulnerability Morphology G*
                   G* → CWE-conditioned local prototype p_{c,k}  →  DP obfuscation  →  Upload

Phase 2 (Server):  {p_{c,k}} k=1..K  →  MCFPA  →  Global prototype bank P*  →  Broadcast

Phase 3 (Client):  Local GNN training with MGMP using P*  →  Detection output
```

### 3.2 Component 1 — Vulnerability-Critical Subgraph Abstraction (VCSA)

**Motivation.** A function's full CPG contains thousands of nodes and edges, most irrelevant to vulnerability patterns. Moreover, concrete API names and variable names prevent cross-project transfer. VCSA solves both problems simultaneously.

**Differentiable Edge Masking.**
For a function's CPG $G = (V, E, X)$ where $X$ are node features (token embeddings), VCSA learns a differentiable edge importance mask $\boldsymbol{\varepsilon} \in [0,1]^{|E|}$ via a lightweight two-layer MLP over edge embeddings. The vulnerability-critical subgraph is:

$$G^* = (V,\ E^* = \{e_i \in E : \varepsilon_i > \tau\},\ X^*)$$

where $\tau$ is a learned threshold. The mask is trained *end-to-end* as part of the vulnerability classifier — not post-hoc like GNNExplainer or PGExplainer.

**Training objective for the mask** — Structural Contrastive Loss:

$$\mathcal{L}_{SCL} = \sum_{(i,j) \in \mathcal{P}^+} \|h_{G^*_i} - h_{G^*_j}\|_2^2
                   + \sum_{(i,j) \in \mathcal{P}^-} \max(0,\ m - \|h_{G^*_i} - h_{G^*_j}\|_2^2)
                   + \lambda_s \cdot \|\boldsymbol{\varepsilon}\|_1$$

where $\mathcal{P}^+$ = same-CWE vulnerable pairs, $\mathcal{P}^-$ = (vulnerable, non-vulnerable) pairs, and $\|\boldsymbol{\varepsilon}\|_1$ encourages sparsity.

**Morphological Abstraction.**
After subgraph extraction, each node label is mapped to a fixed 8-type semantic taxonomy:

| Abstract Type | Covers |
|---|---|
| `MEMORY_ACCESS` | malloc, free, memcpy, buffer reads/writes |
| `ARRAY_INDEX` | array/pointer indexing operations |
| `PTR_DEREF` | pointer dereferences (`*p`, `->`) |
| `CONTROL_BRANCH` | if/else, switch, ternary conditionals |
| `ARITH_OP` | arithmetic operations (+, -, *, /, %) |
| `COMPARISON` | relational operators (<, >, ==, !=) |
| `CALL_SITE` | function/method calls (abstracted) |
| `ASSIGN` | assignment statements |

Mapping is deterministic and derived from AST node types. This produces a compact, project-invariant morphology graph $G^* = (V^*, E^*, \hat{X})$ where $\hat{X}$ contains only abstract type embeddings — no project-specific tokens.

**Novelty vs. prior work:**
- GNNExplainer (Ying et al., NeurIPS 2019): post-hoc, non-differentiable, produces explanations not transferable abstractions.
- PGExplainer (Luo et al., NeurIPS 2020): parametric but still post-hoc explanation, no morphological abstraction.
- VulMorph-Fed VCSA: end-to-end differentiable, produces project-invariant representations designed for cross-project transfer.

### 3.3 Component 2 — Morphology-Conditioned Federated Prototype Aggregation (MCFPA)

**Motivation.** FedAvg averages model parameters — a semantically meaningless operation when clients have heterogeneous CWE distributions. MCFPA replaces parameter sharing with semantic prototype sharing, conditioned on vulnerability type.

**Local Prototype Construction.**
Each client $C_k$ constructs a set of CWE-conditioned prototypes from its local vulnerability morphologies:

$$p_{c,k} = \frac{1}{|\mathcal{D}_{c,k}|} \sum_{(G^*, y) \in \mathcal{D}_{c,k},\ y=1,\ \text{CWE}=c} \text{GNNEnc}(G^*)$$

where $\mathcal{D}_{c,k}$ is the subset of client $k$'s data labelled with CWE type $c$.

**Laplace Differential Privacy at Prototype Level.**
Before uploading, each prototype is obfuscated with calibrated Laplace noise:

$$\tilde{p}_{c,k} = p_{c,k} + \text{Lap}\!\left(0,\ \frac{\Delta f}{\varepsilon}\right)$$

where $\Delta f$ is the global sensitivity of the prototype function, and $\varepsilon$ is the privacy budget. 

**Key advantage over DP-SGD:** The morphological abstraction reduces the effective vocabulary from ~50K API tokens to 8 abstract types, reducing $\Delta f$ by approximately two orders of magnitude. This yields a tighter privacy-utility tradeoff under the same $\varepsilon$.

**CWE-Affinity-Weighted Server Aggregation.**
The server aggregates prototypes using a CWE-affinity weighting matrix rather than sample-count averaging:

$$A_{jk,c} = \text{cosine}\!\left(\tilde{p}_{c,j},\ \tilde{p}_{c,k}\right)$$

$$p^*_c = \frac{\sum_{k} A_{k,c} \cdot \tilde{p}_{c,k}}{\sum_{k} A_{k,c}}, \qquad A_{k,c} = \frac{1}{K-1}\sum_{j \neq k} A_{jk,c}$$

Clients whose local understanding of CWE $c$ aligns with the global consensus contribute more to the global prototype for that CWE type. Clients with no data for CWE $c$ contribute zero. This is semantically aware — not sample-count weighting.

The server broadcasts the **Global Vulnerability Prototype Bank** $\mathcal{P}^* = \{p^*_c\}_{c \in \mathcal{C}}$ to all clients.

**Novelty vs. prior work:**
- FedAvg/FedProx/SCAFFOLD: share model parameters, not semantic prototypes.
- FedProto (Tan et al., AAAI 2022): prototype FL for image/text classification; no code representation, no CWE conditioning, no morphological abstraction.
- VulMorph-Fed MCFPA: first CWE-conditioned prototype aggregation for code vulnerability FL; first prototype-level DP analysis exploiting morphological sensitivity reduction.

### 3.4 Component 3 — Morphology-Guided Message Passing (MGMP)

**Motivation.** Standard GNN message passing aggregates only over local graph neighbors. Cross-project transfer requires a global signal — knowledge of what vulnerability patterns look like across all participating projects. MGMP injects this global signal directly into the message-passing computation.

**Standard message passing (GAT baseline):**

$$h_v^{(l+1)} = \sigma\!\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu}^{(l)} \cdot W^{(l)} h_u^{(l)}\right)$$

**MGMP augmentation:**

$$h_v^{(l+1)} = (1 - \lambda_k) \cdot \underbrace{\sigma\!\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu}^{(l)} \cdot W^{(l)} h_u^{(l)}\right)}_{\text{local neighborhood}} + \lambda_k \cdot \underbrace{\beta_v \cdot \text{ProtoAttn}(h_v^{(l)},\ \mathcal{P}^*)}_{\text{global prototype signal}}$$

where:

$$\text{ProtoAttn}(h_v, \mathcal{P}^*) = \sum_{c} \text{softmax}\!\left(\frac{h_v \cdot p^*_c}{\sqrt{d}}\right) \cdot p^*_c$$

$$\beta_v = \sigma\!\left(W_\beta \cdot [h_v \| \hat{x}_v]\right) \quad \text{(node-level gate based on abstract type } \hat{x}_v\text{)}$$

$$\lambda_k \quad \text{is learned per client via a small meta-network over client data statistics}$$

- $\beta_v$ gates which nodes are influenced by the prototype signal — nodes whose abstract type matches known vulnerability morphologies receive higher $\beta_v$.
- $\lambda_k$ balances local structure vs. global knowledge: data-rich clients rely more on prototypes; isolated clients with few vulnerable samples rely more on local structure.

**Novelty vs. prior work:**
- GAT (Veličković et al., ICLR 2018): local neighborhood attention only; no global signal.
- Cluster-GCN, GraphSAGE: neighbourhood sampling variants; no prototype injection.
- VulMorph-Fed MGMP: first GNN message-passing mechanism augmented with a global semantic prototype bank for cross-project vulnerability knowledge transfer.

### 3.5 Full Algorithm Pseudocode

```
Algorithm: VulMorph-Fed

Input:  K clients with local datasets D_k, global CWE taxonomy C,
        privacy budget ε, rounds T, λ_balance
Output: Global prototype bank P*, local GNN models {f_k}

INITIALISATION
  Each client C_k initialises GNN encoder Enc_k, edge mask MLP M_k,
  local prototype set {p_{c,k}} = ∅

FOR round t = 1 to T:

  ── CLIENT PHASE (parallel, at each C_k) ──────────────────────────────
  1. VCSA: For each (G_i, y_i) in D_k:
     a. Compute edge masks ε_i = M_k(G_i)
     b. Extract G*_i = subgraph(G_i, ε_i > τ)
     c. Apply morphological abstraction: replace node labels → abstract types
     d. Encode: z_i = Enc_k(G*_i)
  
  2. Train M_k and Enc_k jointly:
     L_total = L_BCE(f_k(z_i), y_i)  +  α·L_SCL(z_i)  +  γ·||ε||_1

  3. Local prototype construction:
     For each CWE type c in C:
       p_{c,k} = mean{z_i : y_i=1, CWE(i)=c}  (if |D_{c,k}| > 0, else skip)

  4. Apply Laplace DP:
     p̃_{c,k} = p_{c,k} + Lap(0, Δf/ε)

  5. Upload {p̃_{c,k}}_{c∈C} to server

  ── SERVER PHASE ──────────────────────────────────────────────────────
  6. Compute CWE-affinity matrix: A_{jk,c} = cosine(p̃_{c,j}, p̃_{c,k})
  7. Aggregate: p*_c = Σ_k A_{k,c}·p̃_{c,k} / Σ_k A_{k,c}   for each c
  8. Broadcast P* = {p*_c}_{c∈C} to all clients

  ── CLIENT PHASE (second step) ────────────────────────────────────────
  9. Update MGMP with received P*:
     For each node v in each G*_i:
       h_v^(l+1) = (1-λ_k)·LocalAGG(h_v^(l)) + λ_k·β_v·ProtoAttn(h_v^(l), P*)
  10. Fine-tune local GNN for E_local epochs using updated MGMP

OUTPUT: P* (global prototype bank),  {f_k} (local detection models)
```

### 3.6 Complexity Analysis

| Component | Time complexity | Communication cost |
|---|---|---|
| VCSA (per client) | O(n·|E|·d) per function | — |
| Local prototype construction | O(n·d·|C|) | — |
| Client → server upload | — | O(|C|·d) per round (prototypes only, not weights) |
| MCFPA (server) | O(K²·|C|) | — |
| Server → client broadcast | — | O(|C|·d) per round |
| MGMP (per forward pass) | O(|E|·d + |C|·d) | — |

Communication cost is $O(|C| \cdot d)$ per round — independent of model size — compared to $O(|\theta|)$ for standard FL weight sharing. For $|\mathcal{C}| = 150$ CWE types and $d = 256$, this is approximately 150 KB per round, versus 50–200 MB for sharing full GNN weights.

---

## 4. Evaluation Datasets

### 4.1 Primary Vulnerability Detection Datasets

**Devign**
- Full name: Devign Dataset (FFmpeg + QEMU subset)
- Source: Zhou et al., NeurIPS 2019
- Scale: ~27,318 functions; ~49% vulnerable (manually labelled)
- Language: C
- Projects: FFmpeg, QEMU, Wireshark, Linux kernel
- Download: [https://sites.google.com/view/devign](https://sites.google.com/view/devign) | Mirror: [https://huggingface.co/datasets/DetectVul/devign](https://huggingface.co/datasets/DetectVul/devign)
- Role: Standard GNN baseline; within-project evaluation.

**BigVul**
- Full name: A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries
- Source: Fan et al., MSR 2020
- Scale: ~188,636 functions; 3,754 CVEs; 91 CWE types; 348 GitHub projects
- Language: C / C++
- Download: [https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset)
- Role: Large-scale training; cross-project partitioning by project.

**ReVeal**
- Full name: ReVeal Dataset (Chrome + Debian)
- Source: Chakraborty et al., IEEE TSE 2022
- Scale: ~22,406 functions; ~9.85% vulnerable (realistic class imbalance)
- Language: C / C++
- Projects: Chromium, Debian package pool
- Download: [https://github.com/VulDetProject/ReVeal](https://github.com/VulDetProject/ReVeal)
- Role: Realistic class imbalance; cross-project evaluation from GNN baselines.

**CVEfixes**
- Full name: CVEfixes: Automated Collection of Vulnerabilities and Their Fixes from Open-Source Software
- Source: Bhandari et al., PROMISE 2021
- Scale: v1.0.8 — 11,873 CVEs; 4,249 projects; 272 CWE types; multi-language
- Language: C, C++, Java, Python, PHP, JavaScript (and more)
- Download: [https://zenodo.org/records/13118970](https://zenodo.org/records/13118970) | GitHub: [https://github.com/secureIT-project/CVEfixes](https://github.com/secureIT-project/CVEfixes)
- Role: Multi-language; rich CWE taxonomy for prototype construction; realistic project diversity.

**DiverseVul**
- Full name: DiverseVul: A New Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection
- Source: Chen et al., RAID 2023
- Scale: 18,945 vulnerable + 330,492 non-vulnerable C/C++ functions; 7,514 commits; 150 CWE types; 295+ new projects
- Language: C / C++
- Download: [https://github.com/wagner-group/diversevul](https://github.com/wagner-group/diversevul)
- Role: Largest project diversity; primary dataset for cross-project federated simulation.

**MegaVul**
- Full name: MegaVul: A C/C++ Vulnerability Dataset with Comprehensive Code Representations
- Source: Ni et al., MSR 2024
- Scale: Superset of BigVul + CVEfixes with tree-sitter parsing and rich graph annotations
- Language: C / C++
- Download: [https://github.com/Icyrockton/MegaVul](https://github.com/Icyrockton/MegaVul)
- Role: Pre-extracted graph representations; reduces CPG extraction overhead per client.

**PrimeVul**
- Full name: PrimeVul: A Realistic Dataset for Vulnerability Detection with Code Language Models
- Source: Ding et al., ICSE 2025
- Scale: ~7,000 vulnerable + ~229,000 benign C/C++ functions; paired vuln/patch test set; temporal train/test split
- Language: C / C++
- Download: [https://huggingface.co/datasets/starsofchance/PrimeVul](https://huggingface.co/datasets/starsofchance/PrimeVul)
- Role: Strictest existing benchmark; temporal split prevents data leakage; used to compare against LLM baselines.

**VulGate / VulGate+**
- Full name: VulGate: A Large-Scale, Rigorously Curated Dataset for Software Vulnerability Detection
- Source: arXiv 2508.16625, 2025
- Scale: 236,663 function-level samples; 180 CWE types; VulGate+ includes 500-sample expert-verified cross-project test set
- Language: C / C++
- Composition: Unifies Devign, BigVul, ReVeal, VDISC, D2A, CVEfixes, CrossVul, DiverseVul, PrimeVul, MegaVul
- Download: Available from authors (arXiv 2508.16625)
- Role: Cross-project generalisation evaluation; explicitly designed for this purpose.

### 4.2 Federated Client Simulation Strategy

Since no existing dataset has pre-partitioned federated splits, we simulate $K$ clients by partitioning **DiverseVul** (295+ projects) and **CVEfixes** (4,249 projects) by GitHub repository:

| Split strategy | Description | Purpose |
|---|---|---|
| Random project split | Projects randomly assigned to K clients | IID baseline |
| CWE-stratified split | Each client receives data skewed to 2–3 dominant CWEs | Non-IID simulation |
| Size-stratified split | Clients vary in data volume (power-law distribution) | Realistic industrial scenario |
| Temporal split | Earlier projects train, later projects test | Time-realistic cross-project evaluation |

We evaluate with $K \in \{5, 10, 20, 50\}$ clients to study scalability.

### 4.3 Supplementary Datasets for Defect Prediction Baselines

**tera-PROMISE / NASA MDP**
- Download: [https://openscience.us/repo](https://openscience.us/repo)
- Role: Classic CPDP baseline; software metrics only; used to compare FL-SVD (VulMorph-Fed) against FL-CPDP (FLR, FedDP) approaches.

---

## 5. Evaluation Metrics

### 5.1 Classification Performance Metrics

**Precision**
$$\text{Precision} = \frac{TP}{TP + FP}$$
Proportion of flagged functions that are truly vulnerable. Critical in production: false alarms waste developer time.

**Recall (Sensitivity)**
$$\text{Recall} = \frac{TP}{TP + FN}$$
Proportion of actual vulnerabilities detected. Critical for security: missed vulnerabilities are exploitable.

**F1-Score**
$$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
Harmonic mean; primary performance metric for imbalanced vulnerability datasets.

**Area Under the ROC Curve (AUC-ROC)**
Threshold-independent discrimination ability; allows comparison across models with different operating points.

**Area Under the Precision-Recall Curve (AUC-PR)**
More informative than AUC-ROC under severe class imbalance (typical in vulnerability datasets where <5% of functions are vulnerable).

**False Positive Rate (FPR)**
$$\text{FPR} = \frac{FP}{FP + TN}$$
Explicitly tracked per RQ1; a high FPR renders a scanner unusable in practice.

### 5.2 Cross-Project Generalisation Metrics

Following PrimeVul (Ding et al., ICSE 2025) and Real-Vul (Daoudi et al., TSE 2024):

**Paired Accuracy (P-C)**
$$\text{P-C} = \frac{\text{\# correctly classified (vulnerable, patched) pairs}}{\text{total pairs}}$$
Applied on PrimeVul Paired test set; requires the model to rank the vulnerable version above the patched version. More stringent than standard F1.

**Cross-Project F1 (CP-F1)**
F1 evaluated exclusively on functions from projects *not* seen during training. Primary metric for RQ1 (transfer learning effectiveness).

**Cross-CWE Generalisation**
F1 evaluated per CWE type, including CWE types seen by <3 clients during training. Measures prototype bank's ability to transfer rare vulnerability knowledge.

### 5.3 Federated Learning Metrics

**Communication Rounds to Target F1 (CR@F1)**
Number of federated rounds needed to reach a target F1 threshold (e.g., F1 = 0.60). Measures convergence efficiency.

**Communication Cost per Round (CCR)**
Total bytes transmitted per round (client uploads + server broadcast). VulMorph-Fed theoretical advantage: $O(|\mathcal{C}| \cdot d) \ll O(|\theta|)$.

**Client-level Performance Variance**
$$\sigma^2_{F1} = \frac{1}{K}\sum_{k=1}^{K}(F1_k - \bar{F1})^2$$
Low variance indicates the federated model generalises uniformly across clients; high variance signals non-IID overfitting.

**Centralised vs. Federated Gap (CFG)**
$$\text{CFG} = F1_{\text{centralised}} - F1_{\text{federated}}$$
Measures the cost of federation. Reported for VulMorph-Fed vs. all FL baselines.

### 5.4 Privacy Metrics

**Privacy Budget (ε)**
Formal differential privacy parameter; reported for each experimental configuration. Evaluated at $\varepsilon \in \{0.1, 0.5, 1.0, 2.0, 5.0, \infty\}$.

**Privacy-Utility Tradeoff Curve**
F1 vs. ε curve; demonstrates VulMorph-Fed's advantage from reduced prototype sensitivity $\Delta f$.

**Membership Inference Attack (MIA) Success Rate**
Attack model attempts to infer whether a specific function was in a client's training set from the shared prototypes. Lower MIA success rate = stronger empirical privacy.

**Gradient Reconstruction Error (GRE)**
Following Zhu et al. (NeurIPS 2019 DLG attack): measures how well an adversary can reconstruct training samples from transmitted updates. Compared: raw gradient sharing vs. prototype sharing vs. DP prototype sharing.

### 5.5 Ablation Study Metrics

Report all metrics above under the following ablation conditions:

| Variant | Description |
|---|---|
| Full VulMorph-Fed | VCSA + MCFPA + MGMP (proposed) |
| w/o VCSA | Replace G* with full CPG + raw token embeddings |
| w/o morphological abstraction | G* extracted but no abstract-type mapping |
| w/o MCFPA | Replace with FedAvg weight averaging |
| w/o CWE-affinity | Replace MCFPA with uniform prototype averaging |
| w/o MGMP | Replace with standard GAT message passing |
| w/o DP | Remove Laplace noise from prototypes |
| Centralised oracle | Full data pooling (privacy upper bound) |
| Local only | No federation, each client trains in isolation |

### 5.6 Statistical Testing

All comparisons reported with:
- Wilcoxon signed-rank test (non-parametric, no normality assumption) at $p < 0.05$
- Cliff's delta effect size
- 5-fold cross-validation across project splits; results averaged over 5 random seeds

---

## 6. Baselines

### 6.1 Centralised GNN Baselines
- **Devign** (Zhou et al., NeurIPS 2019) — GGNN on composite CPG
- **IVDetect** (Li et al., ESEC/FSE 2021) — FA-GCN on PDG with GNNExplainer
- **CPVD** (Zhang et al., IEEE TSE 2023) — GAT + domain adaptation (closest centralised baseline)

### 6.2 Centralised LLM/Transformer Baselines
- **LineVul** (Fu & Tantithamthavorn, MSR 2022) — CodeBERT-based
- **LLMxCPG** (Lekssays et al., 2025) — CPG-guided LLM
- **VulGNN** (Ufuktepe et al., arXiv 2026) — lightweight GNN matched to LLM performance

### 6.3 Federated Baselines
- **FLR** (Yamamoto et al., SANER 2023) — FedAvg on software metrics
- **FedDP** (Wang et al., SMR 2025) — knowledge distillation FL for CPDP
- **VulFL-NLP** (Hu et al., arXiv 2024) — FL with CodeBERT backbone
- **VulFL-GNN** (Hu et al., arXiv 2024) — FL with Devign-style GNN backbone
- **FedProx + GAT** — naive federated baseline combining FedProx with standard GAT

---

## 7. Paper Structure (IST Submission)

| Section | Target length |
|---|---|
| 1. Introduction | 1,500 words |
| 2. Background & Related Work | 2,500 words |
| 3. Problem Formulation | 500 words |
| 4. VulMorph-Fed: Proposed Algorithm | 3,000 words |
| 5. Experimental Setup | 1,000 words |
| 6. Results & Analysis (RQ1–RQ4) | 3,000 words |
| 7. Discussion, Limitations & Threats | 1,000 words |
| 8. Conclusion | 400 words |
| References | ~50 entries |
| **Total** | **~13,000 words** |

---
