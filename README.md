# Physics-Informed Neural Network and Momentum Contrastive Learning for Battery State of Health Estimation

This repository contains the code and experiments for the paper "Physics-Informed Neural Network and Momentum Contrastive Learning for Battery State of Health Estimation".

## Abstract

Estimating the State of health (SoH) of lithium-ion batteries is essential for ensuring their safe and efficient operation across various applications. Traditional approaches often struggle to balance accuracy, physical consistency and data efficiency. This paper proposes a novel combination model of Physics-Informed Neural Network and Momentum Contrastive Learning for Battery State of Health Estimation that associates the interpretability of physics-based model with the representational power of contrastive learning. Our innovation lies in developing a unified optimization strategy that carefully balances an estimation physics-informed architecture and the power of contrastive learning. To specifically improve the physics-informed network, we leverage a shared feature encoder to improve representation learning for accurate SoH estimation. For contrastive learning, we design a physics-guided data augmentation strategy with a shared encoder, which generates realistic variations of battery degradation patterns and a momentum encoder architecture, which stabilizes the learning process. Extensive experiments on the NASA lithium-ion battery datasets demonstrate that our model achieves superior performance over state-of-the-art baselines such CNN, BPINN, Informer and XGBoost-ARIMA, achieving a mean absolute error (MAE) average of 0.095% and a root mean squared error (RMSE) average of 0.117% across all batteries. The associations of physics constraints with contrastive learning improve prediction accuracy and enhance model generalization across different battery types and operating conditions, addressing key limitations in existing battery health estimation approaches.

## Model Architecture

Our model combines a Physics-Informed Neural Network-based Estimation (PINNE) component with a Momentum Contrastive Learning component.

![Architecture of Physics-Informed Neural Network-based Estimation and Momentum Contrastive Learning for SoH estimation.](images/Figure_1.png)
*Figure 1: Architecture of Physics-Informed Neural Network-based Estimation and Momentum Contrastive Learning for SoH estimation. (a) Physics-Informed Neural Network-based Estimation incorporating degradation and constraints. (b) Momentum Contrastive Learning with physics-guided data augmentation, momentum encoder, projection heads, and queue-based memory bank.*

![Architecture of the shared feature encoder and estimator.](images/Figure_2.png)
*Figure 2: Architecture of the shared feature encoder and estimator.*

## Data

We use the NASA Ames Prognostics Center of Excellence (PCoE) battery dataset. The following images show features from the B0018 battery.

![B0018 battery time series data features for all cycles.](images/Figure_3.png)
*Figure 3: B0018 battery time series data features for all cycles.*

![B0018 battery time series data features for the first cycle.](images/Figure_4.png)
*Figure 4: B0018 battery time series data features for the first cycle.*

## Results

Our model outperforms several baseline models in SoH estimation.

![Comparison of original and physically-augmented battery  State of Health (SoH) degradation (B0005, B0006, B0007, and B0018).](images/Figure_5.png)
*Figure 5: Comparison of original and physically-augmented battery State of Health (SoH) degradation (B0005, B0006, B0007, and B0018). Original data (solid lines) and augmented data (dotted lines).*

![Battery B0005 SoH estimation results.](images/Figure_6.png)
*Figure 6: Battery B0005 SoH estimation results. Solid blue: our model; dashed red: ground truth.*

### Feature Importance

A feature importance study using Shapley values reveals the most influential features for SoH estimation.

![Shapley value feature importance analysis.](images/Figure_7.png)
*Figure 7: Shapley value feature importance analysis.*

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Other dependencies can be installed from `requierement.txt`:
  ```bash
  pip install -r requierement.txt
  ```

### Training

To train the model, run the main script:

```bash
python benchmark.py
```

## Citation

If you use this work, please cite the original paper.
```
@article{jung2026physics,
  title={Physics-Informed Neural Network and Momentum Contrastive Learning for Battery State of Health Estimation},
  author={Jung, Jiwoo and Bassole, Yipene Cedric Francois and Sung, Yunsick},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```
