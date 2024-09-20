# Relation Extraction Project

This project focuses on the extraction of relational data from text, utilizing two models for the task. Below is an overview of the structure and content of this project repository.

## Project Structure

The project is organized into several key subfolders, each containing specific types of files and serving different roles within the project. Here's a breakdown of each subfolder and its contents:

### BERT

This folder contains files related to the BERT based model implementations for relation extraction.

- `main_bert.ipynb`: A Jupyter notebook that guides through the process of training and evaluating a BERT model for relation extraction.
- `model1.pkl`: A pickle file containing a trained BERT model. Since the size of file is huge we are placing a dummy file. But the original file can be downloaded from:
https://drive.google.com/file/d/1VHohOpGGnZMGY2MoGarkliwxxX9eXUCD/view?usp=share_link


### GNN

This folder is dedicated to Graph Neural Network (GNN) models, particularly focusing on a contrastive learning approach.

- `main_GNN_contrasive.ipynb`: A Jupyter notebook detailing the training and evaluation of a GNN model with a contrastive learning approach.
- `my_trained_model.pth`: A PyTorch file containing a trained GNN model.

### DATA

The TACRED dataset can be downloaded from below link:
https://catalog.ldc.upenn.edu/LDC2018T24
Once the dataset is downloaded the 3 files can be accessed from tacred/data/json

This directory holds the datasets used for training, validating, and testing the models.

- `dev.json`: The development dataset used for fine-tuning the models.
- `test.json`: The test dataset used for evaluating the models.
- `train.json`: The training dataset used to train the models.

### EVALUATION

Contains scripts for evaluating the performance of the models.

- `score.py`: A Python script used to calculate the evaluation metrics of the models based on their predictions.

### INFERENCE

Contains notebooks and modules for performing inference with the trained models.

- `inference.ipynb`: A Jupyter notebook for running inference tasks using the trained models.
- `model1.pkl`: A pickle file containing the BERT-based model, specifically designed for inference.
- `model2.pth`: A PyTorch file containing another GNN-Contrasive model for inference.
- `modules1.py`: A Python module containing utility functions and classes used in `inference.ipynb`.
- `modules2.py`: Another Python module with additional utilities for inference tasks.

## Usage

To use this project, you will need to have Python installed on your machine, along with Jupyter for running the notebooks.

For training and evaluating models, navigate to the respective folders (`BERT` or `GNN`) and open the Jupyter notebooks. Follow the instructions within each notebook. Adjust the dataset paths as necessary based on which models you wish to use for the task.

To run inference, navigate to the `INFERENCE` folder and open the `inference.ipynb` notebook. Adjust the model paths as necessary based on which models you wish to use for your inference tasks.


