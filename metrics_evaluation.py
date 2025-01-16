import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score, top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def process_data(data):
    embeddings = []
    labels = []
    
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                embeddings.append(embedding)
                labels.append(syndrome_id)
    
    return np.array(embeddings), np.array(labels)

def evaluate_model(y_true, y_pred, y_prob, num_classes):
    metrics = {}
    
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        auc = None  
    metrics['AUC'] = auc
    
    f1 = f1_score(y_true, y_pred, average='macro')
    metrics['F1-Score'] = f1
    
    top_5_acc = top_k_accuracy_score(y_true, y_prob, k=5, labels=range(num_classes))
    metrics['Top-5 Accuracy'] = top_5_acc
    
    return metrics

def knn_classification(embeddings, labels, distance_metric='euclidean', k_values=range(1, 16)):
    print(f"Rodando KNN com métrica: {distance_metric}")
    
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    results = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    for k in k_values:
        print(f"Validando para k={k}...")
        knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
        
        y_pred = []
        y_true = []
        y_prob = []
        
        for train_index, test_index in kf.split(embeddings):
            X_train, X_test = embeddings[train_index], embeddings[test_index]
            y_train, y_test = numeric_labels[train_index], numeric_labels[test_index]
            
            knn.fit(X_train, y_train)
            predictions = knn.predict(X_test)
            probabilities = knn.predict_proba(X_test) if hasattr(knn, 'predict_proba') else None
            
            y_pred.extend(predictions)
            y_true.extend(y_test)
            if probabilities is not None:
                y_prob.extend(probabilities)
        

        y_prob = np.array(y_prob)
        metrics = evaluate_model(np.array(y_true), np.array(y_pred), y_prob, num_classes=num_classes)
        metrics['y_true'] = np.array(y_true)  
        metrics['y_prob'] = y_prob           
        results.append((k, metrics))
    
    return results

def plot_roc_curves(results, distance_metric):
    plt.figure(figsize=(10, 6))
    
    for k, metrics in results:
        if 'y_true' in metrics and 'y_prob' in metrics:
            fpr, tpr, _ = roc_curve(metrics['y_true'], metrics['y_prob'][:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'k={k}, AUC={roc_auc:.2f}')
    
    plt.title(f"ROC Curves for KNN ({distance_metric} Distance)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def generate_metrics_table(results, distance_metric):
    table_data = {
        "k": [],
        "AUC": [],
        "F1-Score": [],
        "Top-5 Accuracy": [],
    }
    
    for k, metrics in results:
        table_data["k"].append(k)
        table_data["AUC"].append(metrics['AUC'])
        table_data["F1-Score"].append(metrics['F1-Score'])
        table_data["Top-5 Accuracy"].append(metrics['Top-5 Accuracy'])
    
    df = pd.DataFrame(table_data)
    print(f"\nPerformance Metrics for {distance_metric} Distance:")
    print(df)
    return df

if __name__ == "__main__":
    file_path = "mini_gm_public_v0.1.p"
    
    print("Carregando e processando dados...")
    raw_data = load_pickle_file(file_path)
    embeddings, labels = process_data(raw_data)
    
    print("\nExecutando classificação para Euclidean Distance...")
    results_euclidean = knn_classification(embeddings, labels, distance_metric='euclidean')
    
    print("\nExecutando classificação para Cosine Distance...")
    results_cosine = knn_classification(embeddings, labels, distance_metric='cosine')
    
    print("\nGerando curvas ROC para Euclidean Distance...")
    plot_roc_curves(results_euclidean, "Euclidean")
    
    print("\nGerando curvas ROC para Cosine Distance...")
    plot_roc_curves(results_cosine, "Cosine")
    
    print("\nGerando tabela de métricas para Euclidean Distance...")
    table_euclidean = generate_metrics_table(results_euclidean, "Euclidean")
    
    print("\nGerando tabela de métricas para Cosine Distance...")
    table_cosine = generate_metrics_table(results_cosine, "Cosine")
