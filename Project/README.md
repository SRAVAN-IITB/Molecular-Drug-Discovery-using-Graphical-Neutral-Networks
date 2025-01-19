# Lipophilicity Prediction Using Graph Neural Networks

## Objective

The primary goal of this project is to predict the lipophilicity of chemical compounds using Graph Neural Networks (GNNs). Lipophilicity is a critical molecular property influencing a drug’s absorption, distribution, metabolism, excretion, and toxicity (ADMET). Accurately predicting this property is crucial in the early stages of drug discovery to optimize drug-like characteristics and reduce attrition rates.

## Background

Lipophilicity describes the distribution of a compound between a lipophilic (oil-like) and hydrophilic (water-like) environment, often measured as the logarithmic partition coefficient (logP) or distribution coefficient (logD). For pharmaceutical compounds, logD at physiological pH (7.4) is widely used to understand membrane permeability, solubility, and protein binding.

Traditional computational approaches for lipophilicity prediction rely on quantitative structure-activity relationship (QSAR) models, which often use hand-crafted molecular descriptors. However, these methods are limited by the completeness and quality of the descriptors. Graph Neural Networks provide a promising alternative by directly operating on molecular graphs, capturing the structural and relational information of atoms and bonds without requiring explicit feature engineering.

This project aims to leverage the power of GNNs for predicting lipophilicity by representing molecules as graphs where atoms are nodes and bonds are edges. The training is **supervised**, and the **MoleculeNet dataset** with labeled data will be used to train the model. The network will learn hierarchical features from these graphs to predict logD values with high accuracy

### Packages Needed
`math`
 `matplotlib`
 `networkx`
 `numpy`
 `os`
 `pandas`
`pubchempy`
`rdkit`
`sklearn`
 `torch`
`torch_geometric`
### MoleculeNet Dataset

Import data from MoleculeNet and use cheminformatics libraries like RDKit to preprocess the dataset. MoleculeNet can be imported from `torch_geometric.datasets`.

- explore the dataset by printing all the attributes and properties.
- Use pubchempy to get the name and rdkit to draw the molecular structure.


###  Model Architecture Design

Build a GNN model with 2-3 rounds of message passing with __GCN layer__ , __global aggregation__ to get graph level embedding and __classifier__. Apply the dropout technique to prevent overfitting.

Experiment with **no. of layers**, **types of aggregation**. More importantly, add a **skip connection layer** in the model to preserve the central node information. It will increase the accuracy of the model. You can also try other layers learned from the theory, such as **GraphSAGE**.
The goal is to maximize the model's accuracy on the testing dataset.

###  Model Training

- Split the dataset into training and test sets (e.g., 80%-20%) with a batch size of 64.
- Train the GNN model using a regression loss function like Mean Squared Error (MSE).

###  Performance Evaluation

Evaluate the trained model on the test set using metrics such as:

- Coefficient of determination (R²)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
 
Sometimes, the accuracy of the test set may be low due to overtraining; try to split the datasets with less training data (60-40 or 70-30) and check performance.

Visualize the evaluation metrics by plotting a graph of this Metric Score vs Epochs.


###  Report
[Click here](/report.pdf) for the concise report of 1-2 pages, including the following sections:

1. **Model Architecture Summary** :
 Provides an overview of the model architecture.

2. **Test Dataset Performance**:
Reports the highest accuracy achieved on the test dataset.

3. **Performance Plots**:
Includes the following plots:
    - R² Score vs. Epoch
    - RMSE vs. Epoch

