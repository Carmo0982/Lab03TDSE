# Fashion MNIST - Convolutional Neural Network Architecture Analysis

A comprehensive deep learning project focused on designing, implementing, and analyzing custom Convolutional Neural Network (CNN) architectures for the Fashion MNIST dataset. This project emphasizes intentional architectural decisions, parameter efficiency, and thorough exploratory data analysis to achieve robust image classification performance.

## Getting Started

These instructions will guide you through setting up and running the project on your local machine for development and experimentation purposes. The project includes exploratory data analysis, baseline models, and custom CNN architectures implemented from scratch.

### Prerequisites

To run this project, you'll need the following software and libraries installed:

```bash
Python 3.8 or higher
TensorFlow 2.x
NumPy
Pandas
Matplotlib
Scikit-learn
Jupyter Notebook or JupyterLab
```

### Installing

Follow these step-by-step instructions to set up your development environment:

**Step 1: Clone or download the repository**

```bash
# Navigate to your desired directory
cd /path/to/your/projects
```

**Step 2: Set up a virtual environment (recommended)**

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**Step 3: Install required dependencies**

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter
```

**Step 4: Download the Fashion MNIST dataset**

The dataset files are located in the `archive/` directory and include:

- `fashion-mnist_train.csv`
- `fashion-mnist_test.csv`

These files should be present in the archive folder for the notebooks to run properly.

**Step 5: Launch Jupyter Notebook**

```bash
jupyter notebook
```

Now you can open and run any of the available notebooks to explore the project.

## Running the Project

The project consists of three main Jupyter notebooks that should be executed in the following order:

### 1. Exploratory Data Analysis (EDA)

Open `fashion_mnist_eda.ipynb` to explore:

- Dataset structure and statistics
- Class distribution visualization
- Sample image visualization
- Pixel intensity analysis
- Data preprocessing insights

### 2. Baseline Model

Open `fashion_mnist_baseline.ipynb` to:

- Establish baseline performance with a simple neural network
- Understand the minimum accuracy threshold
- Compare against more complex architectures

### 3. Custom CNN Architecture

Open `fashion_mnist_cnn_custom.ipynb` to:

- Implement a custom-designed CNN architecture
- Understand layer-by-layer design justifications
- Train the model with optimized hyperparameters
- Evaluate performance metrics and visualize results
- Analyze confusion matrices and model predictions

Each notebook is fully documented with explanations and can be run cell-by-cell.

## Model Architecture

The custom CNN architecture consists of:

- **3 Convolutional Layers** with 32, 64, and 128 filters respectively
- **MaxPooling Layers** for spatial dimension reduction
- **Dropout Layers** for regularization
- **Dense Layers** for classification
- **Total Parameters**: Approximately 1.2M parameters (optimized for Fashion MNIST)

Key design principles:

- Hierarchical feature learning (low-level to high-level patterns)
- Parameter efficiency to prevent overfitting
- Intentional architectural choices based on dataset characteristics
- Spatial feature preservation through appropriate padding

## Deployment

**Note on AWS SageMaker Deployment:**

Due to insufficient permissions on AWS SageMaker, automated deployment to a live cloud environment could not be completed. However, the model has been successfully trained and evaluated locally, achieving robust performance metrics.

**Evidence of SageMaker Integration:**

![Notebooks en SageMaker](<img/Screenshot 2026-02-10 204326.png>)

For local deployment, the trained model is saved as `fashion_mnist_cnn_custom.h5` and can be loaded using:

```python
from tensorflow import keras
model = keras.models.load_model('fashion_mnist_cnn_custom.h5')
```

## Built With

- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework for model development
- [Keras](https://keras.io/) - High-level neural networks API
- [NumPy](https://numpy.org/) - Numerical computing library
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [Matplotlib](https://matplotlib.org/) - Data visualization library
- [Scikit-learn](https://scikit-learn.org/) - Machine learning utilities and preprocessing
- [Jupyter](https://jupyter.org/) - Interactive computing environment

## Dataset

This project uses the **Fashion MNIST dataset**, which consists of:

- **60,000 training images**
- **10,000 test images**
- **10 classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Image size**: 28x28 grayscale pixels

Dataset source: [Fashion MNIST (Zalando Research)](https://github.com/zalandoresearch/fashion-mnist)

## Results

The custom CNN architecture achieves:

- **Test Accuracy**: ~90-92%
- **Training Time**: Approximately 15-20 minutes on GPU
- **Model Size**: ~4.8 MB (H5 format)

Performance metrics include:

- Confusion matrix analysis
- Per-class accuracy evaluation
- Training/validation loss curves
- Learning rate optimization results

## Authors

- **Santiago** - _Initial work and architecture design_

## License

This project is developed as part of an academic course (TDS) and is intended for educational purposes.

## Acknowledgments

- Fashion MNIST dataset provided by Zalando Research
- TensorFlow and Keras documentation for implementation guidance
- Academic advisors and course instructors for project requirements
- The deep learning community for design pattern insights
