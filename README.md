# Syndrome-Embedding-KNN-Analysis
**This project uses a set of Python files to perform syndrome classification from embeddings stored in Pickle files. The main flow includes preprocessing, data exploration, classification using K-Nearest Neighbors (KNN), and evaluation of metrics such as AUC, F1-Score, and Top-5 Accuracy.**

Features

Data Preprocessing:
- Loading data from Pickle files.
- Normalization and division into training, validation and test sets.
- Exploratory data analysis.
  
Classification with KNN:
- Implementation of KNN with different distance metrics (Euclidean, Cosine).
- Cross validation for optimization of the k parameter.

Visualization:
- Graphs of metrics (AUC, F1-Score, Top-5 Accuracy) for performance analysis.
- Visualization of embeddings with t-SNE.

Evaluation:
- Generation of metrics tables.
- ROC graphs for model analysis.

-----------------------------------------------------------------------------------------------------------------------------------
To Run the Project: Run the main files depending on the desired functionality:

data_processing.py for data preprocessing.
classification.py for training and validating the KNN model.
data_visu.py for visualizing the embeddings.
metrics_evaluation.py for detailed evaluation of the metrics.

Inputs and Outputs:

Input: Pickle file containing embeddings organized by syndromes.
Output: Metrics, graphs and tables generated at runtime.



├── classification.py          # KNN Classification

├── data_processing.py         # Data preprocessing

├── data_visu.py               # Data visualization

├── metrics_evaluation.py      # Metrics evaluation


