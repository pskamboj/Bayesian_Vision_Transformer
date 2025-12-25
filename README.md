
# Bayesian Vision Transformer for Alzheimer’s Disease Classification

A probabilistic deep learning framework for multi-class Alzheimer’s Disease (AD) stage classification from MRI images using a Bayesian Vision Transformer (BayesianViT) with uncertainty estimation.

---

## Project Overview

This project presents a Bayesian Vision Transformer–based architecture to classify Alzheimer’s disease into multiple stages: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented. Unlike conventional deterministic convolutional neural networks, this model incorporates Bayesian layers that model weights as probability distributions, enabling robust learning and uncertainty-aware predictions.

The system is designed to handle class imbalance, provide calibrated confidence scores, and offer improved reliability for medical decision support.

---

## Key Features

* Bayesian Vision Transformer (ViT-Tiny architecture)
* Epistemic uncertainty estimation using Monte Carlo sampling
* Weighted Random Sampler for class imbalance handling
* KL divergence regularization with annealing
* Progressive K-fold training strategy
* Balanced accuracy and robust evaluation metrics

---

## Model Architecture

* Input image size: 224 × 224
* Patch size: 16 × 16
* Embedding dimension: 256
* Transformer depth: 6 blocks
* Attention heads: 4
* Bayesian layers:

  * Bayesian Linear layers
  * Bayesian Multi-Head Self-Attention
  * Bayesian MLP blocks
* Final classifier: Bayesian Linear head

All deterministic linear layers are replaced with Bayesian layers, allowing probabilistic inference and uncertainty modeling.

---

## Dataset

* Source: Public Alzheimer’s MRI datasets (Kaggle)
* Classes:

  * 0: Non-Demented
  * 1: Very Mild Demented
  * 2: Mild Demented
  * 3: Moderate Demented
* Class imbalance addressed using inverse-frequency sampling

---

## Training Strategy

* Loss function:

  ```
  Total Loss = CrossEntropy Loss + KL_WEIGHT × KL Divergence
  ```
* KL annealing to stabilize Bayesian training
* Optimizer: AdamW
* Cross-validation: Stratified K-Fold
* Typical training duration: 60–80 epochs

---

## Inference and Uncertainty Estimation

During inference, the model performs multiple stochastic forward passes with sampled Bayesian weights. The final prediction is obtained by averaging softmax probabilities across runs, and uncertainty is quantified using prediction variance.

This allows identification of low-confidence predictions, which is critical for medical applications.

---

## Installation

```bash
pip install torch torchvision numpy pandas scikit-learn opencv-python tqdm
```

---

## Usage

Train the model:

```bash
python train.py
```

Run inference on test data:

```bash
python test.py --model_path best_model.pth --data_csv test.csv
```

---

## Project Structure

```
├── train.py
├── test.py
├── model.py
├── dataset.py
├── checkpoints/
├── README.md
```

---

## Results

* Achieved approximately 95 percent validation accuracy on balanced datasets
* Bayesian uncertainty correlates with ambiguous MRI cases
* KL divergence stabilizes with annealing, indicating well-regularized posterior learning

---

## Comparison with Conventional Models

| Feature                       | CNN      | Standard ViT | Bayesian ViT (This Work) |
| ----------------------------- | -------- | ------------ | ------------------------ |
| Global Context Modeling       | No       | Yes          | Yes                      |
| Uncertainty Estimation        | No       | No           | Yes                      |
| Robustness to Imbalanced Data | Limited  | Limited      | High                     |
| Medical Reliability           | Moderate | Moderate     | High                     |

---

## Applications

* Clinical decision support systems
* Early Alzheimer’s disease screening
* Medical image analysis research
* Uncertainty-aware diagnostic tools

---

## Author

Prabhjot Singh
M.Tech, Computational Biology

---
