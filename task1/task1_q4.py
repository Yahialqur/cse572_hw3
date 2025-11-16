import numpy as np
import pandas as pd
from my_kmeans import MyKMeans
import time

def test(data_normalized, K, stop_mode, max_iter, metrics):
    print(f"\n{'='*40}")
    print(f"Testing K means with stop_mode='{stop_mode}' and max_iterations={max_iter}")
    print(f"{'='*40}")
    
    results = {}
    for metric in metrics:
        np.random.seed(42)
        kmeans_model = MyKMeans(k=K, algorithm=metric, max_iterations=max_iter, stop_mode=stop_mode)
        start_time = time.time()
        kmeans_model.fit(data_normalized)
        end_time = time.time()
        total_time = end_time - start_time
        
        final_sse = kmeans_model.compute_sse(data_normalized)
        
        print(f" {metric.capitalize()} - Time: {total_time:.4f} sec, Iterations: {kmeans_model.n_iter_}, Stop Reason: {kmeans_model.stop_reason_}, Final SSE: {final_sse:.4f}")
        results[metric] = {
            'time': total_time,
            'iterations': kmeans_model.n_iter_,
            'stop_reason': kmeans_model.stop_reason_,
            'final_sse': final_sse
        }
    return results

def main():
    data = pd.read_csv('data/data.csv', header=None)
    label = pd.read_csv('data/label.csv', header=None)

    # Normalize data
    data_normalized = data.values / 255.0
    y_true = label.values.ravel() 
    
    K = len(np.unique(y_true))
    MAX_ITERATIONS = 500 

    print(f"Number of clusters K: {K}")
    print(f"Number of samples: {len(data_normalized)}")
    print(f"Max iterations set to: {MAX_ITERATIONS}")

    
    metrics = ['euclidean', 'cosine', 'jaccard']
    
    sse_centroid_stable = test(data_normalized, K, stop_mode='centroid_stable', max_iter=MAX_ITERATIONS, metrics=metrics)
    sse_increase = test(data_normalized, K, stop_mode='sse_increase', max_iter=MAX_ITERATIONS, metrics=metrics)
    max_iterations = test(data_normalized, K, stop_mode='max_iterations', max_iter=MAX_ITERATIONS, metrics=metrics)
    
    print("\n\n" + "="*40)
    print("Summary of Results")
    print("="*40)
    print(f"{'Metric':<10} | {'SSE (Stop: Stable)':<20} | {'SSE (Stop: SSE Inc.)':<20} | {'SSE (Stop: Max Iter 100)':<20}")
    print("-"*80)
    for metric in metrics:
        sse1 = sse_centroid_stable[metric]['final_sse']
        sse2 = sse_increase[metric]['final_sse']
        sse3 = max_iterations[metric]['final_sse']
        print(f"{metric.capitalize():<10} | {sse1:<20.4f} | {sse2:<20.4f} | {sse3:<20.4f}")
    
if __name__ == "__main__":
    main()