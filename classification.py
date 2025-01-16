import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score, top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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

def plot_metrics(results, metric_name, distance_metric):
    ks = [k for k, _ in results]
    metric_values = [metrics[metric_name] for _, metrics in results]
    
    plt.plot(ks, metric_values, label=metric_name)
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs k ({distance_metric} Distance)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path = "mini_gm_public_v0.1.p" 
    
    print("Carregando e processando dados...")
    raw_data = load_pickle_file(file_path)
    embeddings, labels = process_data(raw_data)
    
    results_euclidean = knn_classification(embeddings, labels, distance_metric='euclidean')
    print("Resultados para métrica Euclidean:")
    for k, metrics in results_euclidean:
        print(f"k={k}, AUC={metrics['AUC']}, F1-Score={metrics['F1-Score']}, Top-5 Accuracy={metrics['Top-5 Accuracy']}")
    
    plot_metrics(results_euclidean, "AUC", "Euclidean")
    plot_metrics(results_euclidean, "F1-Score", "Euclidean")
    plot_metrics(results_euclidean, "Top-5 Accuracy", "Euclidean")
    
    results_cosine = knn_classification(embeddings, labels, distance_metric='cosine')
    print("Resultados para métrica Cosine:")
    for k, metrics in results_cosine:
        print(f"k={k}, AUC={metrics['AUC']}, F1-Score={metrics['F1-Score']}, Top-5 Accuracy={metrics['Top-5 Accuracy']}")
    
    plot_metrics(results_cosine, "AUC", "Cosine")
    plot_metrics(results_cosine, "F1-Score", "Cosine")
    plot_metrics(results_cosine, "Top-5 Accuracy", "Cosine")
