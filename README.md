# Adaptive Vision Transformer: Dynamic Patch Weighting

This repository implements **RLViT**, a Vision Transformer enhanced with a reinforcement learning–based patch weighting agent for improved image classification and interpretability.

## Overview

Standard Vision Transformers (ViTs) process all image patches equally, which can introduce noise and inefficiency when only a subset of regions is relevant for classification. RLViT addresses this limitation by training a lightweight policy network to assign importance weights to each patch dynamically, focusing computation on the most informative areas.

Key features:

* **Reinforcement-Learning Patch Weighting**: A policy network learns to highlight salient patches.
* **Flexible ViT Backbone**: Leverages a pretrained ViT to extract patch embeddings, with minimal changes to the core transformer.
* **Lightweight and Interpretable**: Produces patch-weight maps that explain model focus, suitable for domains like medical imaging.
* **Competitive Performance**: Achieves \~97% accuracy on CIFAR-10 in just 3 training epochs without extra data.

## Repository Structure

```
├── research.ipynb         # Main code: model definition, training loop, policy network integration, evaluations and patch weights visualization.
├── visualizer.ipynb       # Basic visualizations of CIFAR-10
├── requirements.txt       # Python dependencies
└── README.md              # Project overview and instructions
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Aayush-prog/Adaptive-Vision-Transformer.git
   cd rlvit
   ```
2. Create a Python environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

### Training and Evaluation

Open `research.ipynb` to:

1. Load and preprocess the CIFAR-10 dataset.
2. Instantiate the RLViT model.
3. Train the agent for a specified number of epochs.
4. Evaluate on the test split and log accuracy.
5. Visualize the learned weights.

### Visualization

Open `visualizer.ipynb` to generate:

* Class distribution of CIFAR-10.
* Sample images of CIFAR-10.

## Dependencies

* Python 3.8+
* PyTorch
* Transformers (Hugging Face)
* Matplotlib
* Seaborn
* Tensorflow

## Results

| Model     | Accuracy  | Epochs     | Extra Data    |
| --------- | --------- | ---------- | ------------- |
| ResNet    | 94.4%     | N/A        | None          |
| ViT-H/14  | 99.5%     | Pretrained | ImageNet, JFT |
| **RLViT** | **97.0%** | 3          | Weights Only  |


---

*Note: For full details on the architecture and training process, refer to `research.ipynb`. Basic visualizations are provided in `visualizer.ipynb`.*
