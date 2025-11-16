import numpy as np
import pandas as pd
from my_kmeans import MyKMeans
import time

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

    
    results = {}
    metrics = ['euclidean', 'cosine', 'jaccard']
        
    # Store results here
    results = {}
    
    for metric in metrics:
        print(f"\n{'='*40}")
        print(f"Running K means with {metric} distance")
        print(f"{'='*40}")
        
        # Initialize model
        kmeans_model = MyKMeans(k=K, algorithm=metric, max_iterations=MAX_ITERATIONS)
        
        # Time and fit model
        start_time = time.time()
        kmeans_model.fit(data_normalized)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Get results 
        iterations_taken = kmeans_model.n_iter_
        stop_reason = kmeans_model.stop_reason_
        final_sse = kmeans_model.compute_sse(data_normalized)
        
        print(f"   Converged in: {total_time:.4f} seconds")
        print(f"   Iterations:   {iterations_taken}")
        print(f"   Stop Reason:  {stop_reason}")
        print(f"   Final SSE:    {final_sse:.4f}")
        
        # Save result for each metric
        results[metric] = {
            'time': total_time,
            'iterations': iterations_taken
        }

    print("\n" + "="*40)
    print("Results Comparison")
    print("="*40)
    print(f"{'Metric':<12} | {'Time (sec)':<15} | {'Iterations':<10}")
    print("-"*40)
    
    for metric, res in results.items():
        print(f"{metric.capitalize():<12} | {res['time']:<15.4f} | {res['iterations']:<10}")

    slowest_time_metric = max(results, key=lambda m: results[m]['time'])
    most_iterations_metric = max(results, key=lambda m: results[m]['iterations'])

    print(f"\nMethod requiring most time: {slowest_time_metric.capitalize()} ({results[slowest_time_metric]['time']:.4f} sec)")
    print(f"Method requiring most iterations: {most_iterations_metric.capitalize()} ({results[most_iterations_metric]['iterations']} iterations)")

if __name__ == "__main__":
    main()