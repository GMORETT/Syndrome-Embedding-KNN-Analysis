import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import collections
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

def split_data(embeddings, labels, test_size=0.2, validation_size=0.1):
    X_train, X_temp, y_train, y_temp = train_test_split(
        embeddings, labels, test_size=test_size + validation_size, stratify=labels, random_state=42
    )
    validation_ratio = validation_size / (test_size + validation_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=validation_ratio, stratify=y_temp, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def normalize_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test

def preprocess_pipeline(file_path):
    print("Carregando arquivo...")
    raw_data = load_pickle_file(file_path)
    
    print("Processando dados...")
    embeddings, labels = process_data(raw_data)
    
    print("Dividindo os dados...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(embeddings, labels)
    
    print("Normalizando os dados...")
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)
    
    print("Pré-processamento concluído.")
    return X_train, X_val, X_test, y_train, y_val, y_test

def exploratory_data_analysis(labels):
    # Contar número de imagens por síndrome
    syndrome_counts = collections.Counter(labels)
    
    # Número total de síndromes
    num_syndromes = len(syndrome_counts)
    
    total_images = len(labels)
    max_images = max(syndrome_counts.values())
    min_images = min(syndrome_counts.values())
    
    print("\nExploratory Data Analysis:")
    print(f"Total Syndromes: {num_syndromes}")
    print(f"Total Images: {total_images}")
    print(f"Max Images per Syndrome: {max_images}")
    print(f"Min Images per Syndrome: {min_images}")
    
    plt.figure(figsize=(12, 6))
    plt.bar(syndrome_counts.keys(), syndrome_counts.values(), alpha=0.75)
    plt.xlabel("Syndrome ID")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Images per Syndrome")
    plt.show()

if __name__ == "__main__":
    file_path = "mini_gm_public_v0.1.p"
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_pipeline(file_path)
    
    print("Tamanhos dos conjuntos:")
    print(f"Treino: {X_train.shape}, {y_train.shape}")
    print(f"Validação: {X_val.shape}, {y_val.shape}")
    print(f"Teste: {X_test.shape}, {y_test.shape}")
    
    exploratory_data_analysis(y_train)
