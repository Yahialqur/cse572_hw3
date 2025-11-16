from my_kmeans import MyKMeans
import numpy as np
import pandas as pd

def main():
    # Load dataset
    data = pd.read_csv('data/data.csv')
    label = pd.read_csv('data/label.csv')
    
    data_normalized = data.values / 255.0
    
    K = len(np.unique(label))
    print(f"Number of clusters K: {K}")
    print(f"Number of samples: {len(data)}")
    
    metrics = ['euclidean', 'cosine', 'jaccard']
    
    print("\n"+"="*40)
    print("Running K means with euclidean distance")
    print("="*40)
    kmeans_euclidean = MyKMeans(k=K, algorithm='euclidean', max_iterations=100)
    kmeans_euclidean.fit(data_normalized)
    print("Centroids:\n", kmeans_euclidean.centroids)
    print("Labels:\n", kmeans_euclidean.labels_)
    print("SSE:", kmeans_euclidean.compute_sse(data_normalized))
    
    print("\n"+"="*40)
    print("Running K means with cosine distance")
    print("="*40)
    kmeans_cosine = MyKMeans(k=K, algorithm='cosine', max_iterations=100)
    kmeans_cosine.fit(data_normalized)
    print("Centroids:\n", kmeans_cosine.centroids)
    print("Labels:\n", kmeans_cosine.labels_)
    print("SSE:", kmeans_cosine.compute_sse(data_normalized))   
    
    print("\n"+"="*40)
    print("Running K means with jaccard distance")
    print("="*40)
    kmeans_jaccard = MyKMeans(k=K, algorithm='jaccard', max_iterations=100)
    kmeans_jaccard.fit(data_normalized)
    print("Centroids:\n", kmeans_jaccard.centroids)
    print("Labels:\n", kmeans_jaccard.labels_)
    print("SSE:", kmeans_jaccard.compute_sse(data_normalized))
    
if __name__ == "__main__":
    main()