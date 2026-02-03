# Lab 03 - Convolutional Architecture Analysis and Design

## Description

Laboratory focused on the study and design of convolutional neural networks (CNNs). The goal is to understand how convolutional layers introduce inductive bias into learning systems, and how architectural decisions affect model performance, scalability, and interpretability.

## Learning Objectives

- Understand the role and mathematical intuition behind convolutional layers
- Analyze how architectural decisions (kernel size, depth, stride, padding) affect learning
- Compare convolutional layers with fully connected layers for image-like data
- Perform exploratory data analysis (EDA) for neural network tasks
- Communicate architectural and experimental decisions clearly

## Dataset

**Name**: [Specify selected dataset]

**Source**: [TensorFlow Datasets / PyTorch / Kaggle]

**Justification**: [Explain why it is appropriate for convolutional layers]

**Characteristics**:

- Dataset size: [# samples]
- Classes: [# classes]
- Image dimensions: [height x width x channels]
- Preprocessing applied: [normalization, resizing, etc.]

## Project Structure

### 1. Data Exploration (EDA)

- Class distribution
- Dimension and image analysis
- Sample visualization per class

### 2. Baseline Model (Non-Convolutional)

**Architecture**: [Flatten + Dense layers]

**Parameters**: [# total parameters]

**Performance**:

- Training accuracy: [%]
- Validation accuracy: [%]

**Observed limitations**: [Describe]

### 3. Convolutional Architecture

**Design**:

- Number of convolutional layers: [#]
- Kernel sizes: [e.g., 3×3]
- Stride and padding: [values]
- Activation functions: [ReLU, etc.]
- Pooling strategy: [MaxPooling, etc.]

**Design justification**: [Explain decisions]

**Parameters**: [# total parameters]

### 4. Controlled Experiments

**Explored aspect**: [Kernel size / # filters / Depth / Pooling / Stride]

**Results**:

| Configuration | Accuracy | Loss | # Parameters |
| ------------- | -------- | ---- | ------------ |
| Config 1      | [%]      | [#]  | [#]          |
| Config 2      | [%]      | [#]  | [#]          |
| Config 3      | [%]      | [#]  | [#]          |

**Qualitative observations**: [Describe trade-offs]

### 5. Interpretation and Architectural Reasoning

**Why did convolutional layers outperform (or not) the baseline?**
[Answer]

**What inductive bias does convolution introduce?**
[Answer]

**In what type of problems would convolution NOT be appropriate?**
[Answer]

### 6. SageMaker Deployment

- **Endpoint**: [URL or endpoint name]
- **Status**: [Active/Inactive]
- **Inference**: [Description of how to perform inferences]

## Main Results

### Final Comparison

| Model    | Accuracy | Loss | Parameters | Training time |
| -------- | -------- | ---- | ---------- | ------------- |
| Baseline | [%]      | [#]  | [#]        | [#]           |
| CNN      | [%]      | [#]  | [#]        | [#]           |

### Conclusions

[Summary of main findings]

## Requirements

```bash
pip install tensorflow torch torchvision numpy pandas matplotlib seaborn jupyter sagemaker boto3
```

## Usage

### Local Training

```bash
jupyter notebook Lab03_CNN.ipynb
```

### SageMaker Deployment

[Deployment instructions]

## File Structure

```
Lab03TDSE/
├── README.md
├── Lab03_CNN.ipynb
├── data/
│   └── [dataset files]
├── models/
│   ├── baseline_model.h5
│   └── cnn_model.h5
├── results/
│   ├── training_plots.png
│   └── confusion_matrix.png
└── sagemaker/
    └── deployment_script.py
```

## References

- [Paper or resource 1]
- [Paper or resource 2]
- Dataset: [Link to dataset]

## Author

Santiago - TDSE - First Term 2026

## Important Notes

- This is not a hyperparameter tuning exercise
- Copy-pasted architectures without justification will receive low scores
- Code correctness matters less than architectural reasoning
