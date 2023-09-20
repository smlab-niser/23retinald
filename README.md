# Retinal Fundus Multi-Disease Image Classification using Hybrid CNN-Transformer-Ensemble Architectures

Our project focuses on advancing the field of medical image classification, particularly in the context of retinal fundus images. We aim to develop a state-of-the-art system for the accurate and efficient classification of multi-disease patterns within these images. The project leverages the power of hybrid architectures, combining Convolutional Neural Networks (CNNs) with Transformers, and employs an ensemble approach to enhance classification performance.

Retinal fundus images contain vital diagnostic information for various eye diseases, including diabetic retinopathy, glaucoma, and age-related macular degeneration. Accurate and timely classification of these diseases from images can aid in early detection and intervention, potentially preventing vision loss.

Our approach integrates CNNs, known for their excellence in image feature extraction, with Transformers, renowned for their sequence modeling capabilities. This hybrid architecture enables the model to capture both local and global image features, enhancing its ability to discriminate between different disease patterns.

Furthermore, we employ an ensemble of models to improve the robustness and generalization of the classification system. This ensemble approach combines the strengths of multiple models to make more reliable predictions, even in the presence of noisy or challenging data.

Through our project, we aim to contribute to the advancement of medical image analysis, particularly in the critical domain of retinal fundus image classification. Our hybrid CNN-Transformer-Ensemble architectures hold the potential to provide accurate diagnoses, ultimately improving the quality of eye care and patient outcomes.

## Dataset

We used the [Multi-Label Retinal Diseases (MuReD) Dataset](https://data.mendeley.com/datasets/pc4mb3h8hz/2) for our research. This dataset contains 2208 retinal fundus images for multi label classification on 20 disease labels, divided into 1766 for training and 442 for validation. The dataset was created and used in paper [Multi-Label Retinal Disease Classification using Transformers](https://arxiv.org/abs/2207.02335).

**Citation:**

Mendeley Data (2022). *Multi-Label Retinal Diseases (MuReD) Dataset*. Mendeley. [Dataset Link](https://data.mendeley.com/datasets/pc4mb3h8hz/2)

## Usage

You can access and download the dataset from the provided [Dataset Link](https://data.mendeley.com/datasets/pc4mb3h8hz/2).

```python
# Example code on how to load the dataset using Python
import pandas as pd

dataset_url = "https://data.mendeley.com/datasets/pc4mb3h8hz/2"
data = pd.read_csv(dataset_url)

# Your code for dataset exploration and analysis here
