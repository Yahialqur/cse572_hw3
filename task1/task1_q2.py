import numpy as np
import pandas as pd
from my_kmeans import MyKMeans
import time

def calculate_accuracy(y_true, y_pred_clusters, k):

    cluster_map = {}
    
    # Build the Majority Vote Map
    for i in range(k):
        # Find indices of all data assigned to cluster 'i'
        cluster_indices = np.where(y_pred_clusters == i)[0]
        
        if len(cluster_indices) == 0:
            cluster_map[i] = -1
            continue
            
        # Get true labels for points in this cluster
        true_labels_for_cluster = y_true[cluster_indices]
        
        # Find unique labels and counts in cluster
        unique_labels, counts = np.unique(true_labels_for_cluster, return_counts=True)
        
        # Find index of highest count
        max_count_index = np.argmax(counts)
        
        # Get label with highest count
        majority_label = unique_labels[max_count_index]
        
        # Assign to map
        cluster_map[i] = majority_label
        
    print(f"Cluster Map (Cluster ID -> True Label): {cluster_map}")
    
    # Translate cluster ids to Mapped Labels
    y_pred_mapped = np.zeros_like(y_pred_clusters)
    
    for cluster_id, true_label in cluster_map.items():
        # Find all points assigned to cluster
        mask = (y_pred_clusters == cluster_id)
        # Assign true_label
        y_pred_mapped[mask] = true_label
        
    # Calculate final accuracy 
    correct_predictions = np.sum(y_pred_mapped == y_true)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    return accuracy

def main():
    data = pd.read_csv('data/data.csv', header=None)
    label = pd.read_csv('data/label.csv', header=None)

    # Normalize data
    data_normalized = data.values / 255.0
    y_true = label.values.ravel() 
    
    K = len(np.unique(y_true))
    print(f"Number of clusters K: {K}")
    print(f"Number of samples: {len(data_normalized)}")
    
    results = {}
    metrics = ['euclidean', 'cosine', 'jaccard']
    
    for metric in metrics:
        print(f"\n{'='*40}")
        print(f"Running K means with {metric} distance")
        print(f"{'='*40}")
        
        kmeans_model = MyKMeans(k=K, algorithm=metric, max_iterations=100)
        
        start_time = time.time()
        kmeans_model.fit(data_normalized)
        end_time = time.time()
        
        sse = kmeans_model.compute_sse(data_normalized)
        
        results[metric] = {
            'model': kmeans_model,
            'sse': sse,
            'time': end_time - start_time
        }
        
        print(f"SSE: {sse}")
        print(f"Time: {end_time - start_time:.4f} seconds")

    # SSE Comparison
    print("\n" + "="*40)
    print("SSE Comparison")
    print("="*40)
    for metric, res in results.items():
        print(f"{metric.capitalize()} SSE: {res['sse']:.4f}")
    best_sse_metric = min(results, key=lambda m: results[m]['sse'])
    print(f"\nBest Metric (by SSE): {best_sse_metric.capitalize()} "
          f"({results[best_sse_metric]['sse']:.4f})")

    # Accuracy Comparison
    print("\n" + "="*40)
    print("Accuracy Comparison")
    print("="*40)
    
    accuracies = {}
    
    for metric, res in results.items():
        print(f"\nCalculating accuracy for {metric.capitalize()}...")
        model = res['model']
        
        # Calculate accuracy
        acc = calculate_accuracy(y_true, model.labels_, K)
        accuracies[metric] = acc
        
        print(f"{metric.capitalize()} Accuracy: {acc * 100:.2f}%")
        
    best_acc_metric = max(accuracies, key=accuracies.get)
    print(f"\nBest Metric (by Accuracy): {best_acc_metric.capitalize()} "
          f"({accuracies[best_acc_metric] * 100:.2f}%)")

if __name__ == "__main__":
    main()