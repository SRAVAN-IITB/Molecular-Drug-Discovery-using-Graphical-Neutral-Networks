
### TODO 1: Importing the necessary libraries - numpy, matplotlib and time
import numpy as np
import matplotlib.pyplot as plt
import time

### TODO 2
### Load data from data_path
### Check the input file spice_locations.txt to understand the Data Format
### Return : np array of size Nx2
def load_data(data_path):
    data = np.loadtxt(data_path, delimiter=',')
    return data


### TODO 3.1
### If init_centers is None, initialize the centers by selecting K data points at random without replacement
### Else, use the centers provided in init_centers
### Return : np array of size Kx2
def initialise_centers(data, K, init_centers=None):
    if init_centers is None:
        indices = np.random.choice(data.shape[0], K, replace=False).astype(int)
        return data[indices]
    return init_centers

### TODO 3.2
### Initialize the labels to all ones to size (N,) where N is the number of data points
### Return : np array of size N
def initialise_labels(data):
    return np.full(data.shape[0], 1)

### TODO 4.1 : E step
### For Each data point, find the distance to each center
### Return : np array of size NxK
def calculate_distances(data, centers):
    distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
    return distances
    
### TODO 4.2 : E step
### For Each data point, assign the label of the nearest center
### Return : np array of size N
def update_labels(distances):
    return np.argmin(distances, axis=1)

### TODO 5 : M step
### Update the centers to the mean of the data points assigned to it
### Return : np array of size Kx2
def update_centers(data, labels, K):
    # centers = np.array([np.mean(data[labels == i], axis=0) for i in range(K)])
    centers = np.array([data[labels == k].mean(axis=0) if np.any(labels == k) else data[np.random.choice(data.shape[0])] for k in range(K)])
    return centers

### TODO 6 : Check convergence
### Check if the labels have changed from the previous iteration
### Return : True / False
def check_termination(labels1, labels2):
    return np.array_equal(labels1, labels2)

### simulate the algorithm in the following function. run.py will call this
### function with given inputs.
def kmeans(data_path:str, K:int, init_centers):
    '''
    Input :
        data (type str): path to the file containing the data
        K (type int): number of clusters
        init_centers (type numpy.ndarray): initial centers. shape = (K, 2) or None
    Output :
        centers (type numpy.ndarray): final centers. shape = (K, 2)
        labels (type numpy.ndarray): label of each data point. shape = (N,)
        time (type float): time taken by the algorithm to converge in seconds
    N is the number of data points each of shape (2,)
    '''
    data = load_data(data_path)
    centers = initialise_centers(data, K, init_centers)
    labels = initialise_labels(data)
    start_time = time.perf_counter()
    
    while True:
        distances = calculate_distances(data, centers)
        new_labels = update_labels(distances)
        
        if check_termination(labels, new_labels):
            break
        
        labels = new_labels
        centers = update_centers(data, labels, K)
        
        elapsed_time = time.perf_counter() - start_time
        return centers, labels, elapsed_time

### to visualise the final data points and centers.
def visualise(data_path, labels, centers):
    data = load_data(data_path)

    # Scatter plot of the data points
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    # Adding title and labels
    plt.title('K-means  Clustering')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.show()
    plt.savefig('k_means.png')
    return plt
