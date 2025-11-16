import numpy as np

class MyKMeans():
    def __init__ (self, k, max_iterations, algorithm):
        self.k = k
        self.max_iterations = max_iterations
        self.algorithm = algorithm
        self.iterations = 0
        self.reason_for_stop = ""
        
    def fit(self, data):
        '''
            Train k means algorithm on data
        '''
        # Randomly initialize centroids
        self.centroids = data[np.random.choice(data.shape[0], self.k, replace=False)]
        
        # Set previous state
        prev_sse = np.inf
        prev_centroids = None
        prev_labels = None
        
        # Stop if max iterations reached
        self.stop_reason_ = f"max_iterations ({self.max_iterations}) reached"
        
        # Create clusters by assigning each data point to nearest centroid
        for i in range(self.max_iterations):
            self.labels_ = self._assign_clusters(data, self.algorithm)
            
            current_sse = self.compute_sse(data)
            
            # Stop if SSE increases
            if current_sse > prev_sse:
                self.stop_reason_ = f"SSE increased at iteration {i}"
                self.n_iter_ = i
                # Revert to the previous (better) state
                self.centroids = prev_centroids
                self.labels_ = prev_labels
                break
            
            # Save the current state before updating
            prev_sse = current_sse
            prev_centroids = self.centroids.copy()
            prev_labels = self.labels_.copy()
            
            new_centroids = self._update_centroids(data, self.labels_)
            
            # Stop if centroids do not change
            if np.all(self.centroids == new_centroids):
                self.stop_reason_ = f"Centroids stable at iteration {i+1}"
                self.n_iter_ = i + 1
                self.centroids = new_centroids
                break
            
            self.centroids = new_centroids
            
            if i == self.max_iterations - 1:
                self.n_iter_ = self.max_iterations
            
    def _assign_clusters(self, data, algorithm):
        '''
            Helper function to create clusters
            Assigns each data point to the nearest centroid
        '''
        if algorithm == 'euclidean':
            distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        elif algorithm == 'cosine':
            distances = self._cosine_distance(data, self.centroids)
        elif algorithm == 'jaccard':
            distances = self._jaccard_distance(data, self.centroids)

        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, data, clusters):
        '''
            Helper function to update centroids
            Computes the new centroids as the mean of all data points assigned to each cluster
        '''
        new_centroids = np.zeros((self.k, data.shape[1]))  
        for k in range(self.k):
            cluster_points = data[clusters == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                # Reinitialize empty cluster to a random data point
                new_centroids[k] = data[np.random.randint(0, len(data))]
        return new_centroids

    def _cosine_distance(self, data, centroids):
        '''
            Compute cosine distance between data points and centroids
            Cosine distance = 1 - cosine similarity
        '''
        # Normalize data and centroids
        data_norm = np.linalg.norm(data, axis=1, keepdims=True)
        centroids_norm = np.linalg.norm(centroids, axis=1, keepdims=True)

        # Avoid division by zero
        data_norm = np.where(data_norm == 0, 1, data_norm)
        centroids_norm = np.where(centroids_norm == 0, 1, centroids_norm)

        # Compute cosine similarity
        cosine_similarity = np.dot(data / data_norm, (centroids / centroids_norm).T)

        # Convert to distance (1 - similarity)
        cosine_distance = 1 - cosine_similarity

        return cosine_distance

    def _jaccard_distance(self, data, centroids):
        '''
            Compute generalized Jaccard distance between data points and centroids
            Generalized Jaccard distance = 1 - J(X,Y)
        '''

        n_samples = data.shape[0]
        k = centroids.shape[0]
        distances = np.zeros((n_samples, k))

        for i in range(k):
            centroid = centroids[i]

            # Compute min and max for each feature
            min_vals = np.minimum(data, centroid)
            max_vals = np.maximum(data, centroid)

            # Sum
            numerator = np.sum(min_vals, axis=1)
            denominator = np.sum(max_vals, axis=1)

            # Avoid division by zero
            denominator = np.where(denominator == 0, 1, denominator)

            # Jaccard coefficient
            jaccard_coef = numerator / denominator

            # Convert to distance
            distances[:, i] = 1 - jaccard_coef

        return distances

    def compute_sse(self, data):
        '''
            Compute the Sum of Squared Error (SSE) for the current clustering
            SSE = sum of squared distances between each point and its assigned centroid
        '''
        if not hasattr(self, 'labels_'):
            self.labels_ = self._assign_clusters(data, self.algorithm)
        
        sse = 0
        if self.algorithm == 'euclidean':
            sse = np.sum((data - self.centroids[self.labels_]) ** 2)
        elif self.algorithm == 'cosine':
            distances = self._cosine_distance(data, self.centroids)
            sse += np.sum(distances[np.arange(len(data)), self.labels_] ** 2)
        elif self.algorithm == 'jaccard':
            distances = self._jaccard_distance(data, self.centroids)
            sse += np.sum(distances[np.arange(len(data)), self.labels_] ** 2)
        
        return sse
