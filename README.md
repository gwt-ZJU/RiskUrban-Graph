# RiskUrban-Graph

Official implementation of **SceneDGF** (Scene-conditioned Dual Graph Fusion) from the paper:

> **SceneDGF: Scene-Conditioned Multimodal Learning for Urban Road Crash Risk Prediction**
> *Under review*

## 📢 Code Release

The source code, pretrained models, and processed data will be **made publicly available upon acceptance of the paper**. Please star or watch this repository to receive updates.

## Overview

SceneDGF is a multimodal deep learning framework for segment-level urban road crash risk prediction. It integrates street-view imagery (SVI), hourly traffic flow, and road-network topology:

- **Scene Encoder**: encodes semantic proportions and spatial co-occurrence of streetscape elements via a semantic adjacency matrix.
- **Traffic Encoder**: captures 24-hour speed and volume dynamics with 1D convolutions and multi-head self-attention.
- **Learnable Graph Fusion (LGF)**: applies FiLM-based scene conditioning to dual-graph convolutions over a spatial flow graph and a spatial similarity graph, fused by a learnable weight.
- **Interpretability**: Integrated Gradients attribution decomposed into self and interaction (co-occurrence) components.

## Planned Contents

- [ ] Model implementation (PyTorch / PyTorch Geometric)
- [ ] Training and evaluation scripts
- [ ] Baseline implementations
- [ ] Attribution and perturbation analysis scripts
- [ ] Processed sample data and data format description

## Data Availability

The raw street-view imagery, license-plate-recognition traffic data, and crash records used in this study cannot be shared publicly due to privacy and confidentiality restrictions. They are available from the corresponding author upon reasonable request.

## Citation

Citation information will be provided after publication.

## Contact

For questions, please contact Sheng Jin (jinsheng@zju.edu.cn) or Wentong Guo (12512016@zju.edu.cn).
