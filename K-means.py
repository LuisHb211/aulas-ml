import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(data, k):
  indices = np.random.choice(data.shape[0], k, replace=False)
  return data[indices]

def assign_clusters(data, centroids):
  distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
  return np.argmin(distances, axis=1)

def update_centroids(data, labels, k):
  new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
  return new_centroids

def calculate_total_error(data, centroids, labels):
  total_error = 0
  for i in range(centroids.shape[0]):
    cluster_points = data[labels == i]
    total_error += np.sum((cluster_points - centroids[i]) ** 2)
  return total_error

def k_means(data, k, max_iters=100, tol=1e-4):
  centroids = initialize_centroids(data, k)
  errors = []  
  for _ in range(max_iters):
    labels = assign_clusters(data, centroids)
    new_centroids = update_centroids(data, labels, k)
    total_error = calculate_total_error(data, centroids, labels)
    errors.append(total_error)
    if np.all(np.abs(new_centroids - centroids) < tol):
      break
    centroids = new_centroids
  return centroids, labels, errors

if __name__ == "__main__":
  data = np.loadtxt('observacoescluster.txt')
  
  k = 3
  centroids, labels, errors = k_means(data, k)
  
  print("Centroids:\n", centroids)
  print("Labels:\n", labels)
  
  plt.plot(range(1, len(errors) + 1), errors, marker='o')
  plt.xlabel('Iteration')
  plt.ylabel('Total Quadratic Error (EQT)')
  plt.title('EQT vs Iteration')
  plt.show()