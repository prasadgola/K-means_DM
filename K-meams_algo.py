import numpy as np
import matplotlib.pyplot as plt

def k_means(data, k, iterations):
  """
  K-means clustering algorithm.

  Args:
    data: The data to be clustered.
    k: The number of clusters.
    iterations: The number of iterations to run the algorithm.

  Returns:
    A list of cluster labels, one for each data point.
  """

  # Initialize the centroids randomly.
  centroids = np.random.randint(0, len(data), (len(data), len(data[0])))

  # Run the algorithm for the specified number of iterations.
  for i in range(iterations):
    # Assign each data point to its nearest centroid.
    distances = np.linalg.norm(data - centroids, axis=1)
    labels = np.argmin(distances, axis=0)

    # Update the centroids.
    for j in range(k):
      centroids[j] = np.mean(data[labels == j].reshape(-1, len(data[0])), axis=0)

  return labels, centroids

def main():
  # Load the data.
  data_file = input("Enter the file name = ")
  data = np.loadtxt(f"UCI_datasets/{data_file}")


#   Run the K-means clustering algorithm for different values of K.
  sse_list = []
  for k in range(2, 11):
    # print(0, len(data), (len(data), len(data[0])))
    labels, centroids = k_means(data, k, 20)
    sse = np.sum((data - centroids[labels])**2)
    print("For k = {} After 20 iterations: SSE error = {}".format(k, sse))
    sse_list.append(sse)

  # Plot the SSE vs k chart.
  plt.plot([2, 3, 4, 5, 6, 7, 8, 9, 10], [sse for sse in sse_list])
  plt.xlabel("K")
  plt.ylabel("SSE")
  plt.show()

if __name__ == "__main__":
  main()


#   data = np.loadtxt(os.path.join("UCI_datasets", "yeast_training.txt"))