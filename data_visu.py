import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

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

def visualize_tsne(embeddings, labels, perplexity=30, learning_rate=200, n_iter=1000):
    print("Executando t-SNE para reduzir a dimensionalidade...")
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    print("Convertendo rótulos para valores numéricos...")
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)
    
    print("Gerando gráfico...")
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], 
        c=numeric_labels, cmap='tab20', alpha=0.7, edgecolors='k'
    )
    plt.colorbar(scatter, label="Syndrome ID (encoded)")
    plt.title("t-SNE Visualization of Embeddings by Syndrome ID")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path = "mini_gm_public_v0.1.p"
    
    print("Carregando e processando dados...")
    raw_data = load_pickle_file(file_path)
    embeddings, labels = process_data(raw_data)
    
    visualize_tsne(embeddings, labels)
