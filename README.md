# Food-101 Image Classification

## Project Overview
This repository contains a deep learning-based image classification project using the **Food-101 dataset**. The goal is to build a robust food classifier by leveraging transfer learning with feature extractors like Inception and ResNet. The project includes exploratory data analysis (EDA), data cleaning, train-test splitting, hyperparameter tuning, and visualization of feature maps. The final model is trained, evaluated, and saved for inference.

---

## Table of Contents
1. [Project Objectives](#project-objectives)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Setup Instructions](#setup-instructions)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Results](#results)
8. [contact](#Contact)

---

## Project Objectives
The objective of this assignment is to develop a deep learning (image classifier) model using the Food-101 dataset with the following operations:
- Use **random 25 classes** out of 101 from the given dataset.
- Perform **Exploratory Data Analysis (EDA)** on the extracted classes.
- Clean the dataset for **wrong labels**.
- Create a **Train/Test Split**.
- Use **3 different feature extractors** (e.g., Inception, ResNet, etc.).
- Add required layers to work with the feature extractor.
- Perform **Hyperparameter tuning** (10-12 experiments with reasoning).
- Plot **feature maps** of the last or second-last section of the finalized classifier.

---

## Dataset
The dataset used in this project is the **Food-101 dataset**, which contains 101 food categories with 101,000 images. For each class:
- **750 training images** (not cleaned, may contain noise or wrong labels).
- **250 manually reviewed test images**.
- All images are rescaled to have a maximum side length of 512 pixels.

Download the dataset from [here](https://www.kaggle.com/datasets/dansbecker/food-101).

---

## Requirements
To run this project, you need the following Python libraries:
- TensorFlow/Keras or PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- Pillow

You can install the required libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Setup Instructions
Clone the repository:
```bash

git clone https://github.com/ChinmayBH/food-classification-computer-vision.git
cd food-101-image-classifier
```

-------
Download the Food-101 dataset and extract it into the data/ directory.

Install the required libraries:
```bash

pip install -r requirements.txt
```

Open the Jupyter Notebook or Python script to run the project.
----
## Usage
Data Preprocessing:
Clean the dataset for wrong labels.
Perform train-test splitting.

Model Training:
Use feature extractors like Inception, ResNet, etc.
Add required layers for classification.
Perform hyperparameter tuning.

Evaluation:
Visualize training and validation metrics (accuracy, loss).
Plot feature maps of the last or second-last section of the classifier.

Inference:
Use the trained model to make predictions on new images.

## Project Structure
```bash

food-101-image-classifier/
│
├── data/                     # Folder for dataset
│   ├── train/                # Training images
│   └── test/                 # Test images
│
├── models/                   # Saved model files
│   └── trained_model.h5      # Trained model file
│
├── notebooks/                # Jupyter notebooks
│   └── food_classification.ipynb  # Main notebook
│
├── scripts/                  # Additional scripts
│   └── preprocess_data.py    # Data preprocessing script
│
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── .gitignore                # Files/folders to ignore in Git
```

## Results
Training and Validation Metrics:
Accuracy and loss plots for each epoch.
Hyperparameter:Summary of experiments and their results.

## Contact
For any questions or feedback, please contact:
Email: Chinmaybhalerao0912@gmail.com

GitHub: https://github.com/ChinmayBH