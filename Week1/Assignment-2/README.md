# K-Means Clustering Algorithm

## **Problem Statement**  
Given a dataset of 2D coordinates representing points of interest, the task is to identify \( K \) central locations that best represent clusters of these points. Using the **K-Means clustering algorithm**, group the points into \( K \) clusters, where each cluster represents a region around a central point. The algorithm iteratively refines the clusters and their centers to minimize the distance between points and their assigned center.

---

## **Input**  
1. A dataset containing 2D coordinates of points (\( N \) points).  
2. \( K \): A guess for the number of clusters.  

---

## **Output**  
1. Coordinates of \( K \) cluster centers.  
2. Cluster labels for each point in the dataset.

---

## **Algorithm**  

### **1. Initialization**  
- Randomly select \( K \) points from the dataset as the initial cluster centers.  
- Assign a label (0 to \( K-1 \)) to each point, indicating its cluster membership.

### **2. Iterative Refinement**  
Repeat the following steps until the cluster assignments (labels) do not change:  

#### **E-Step (Assignment)**  
1. For each point, calculate the Euclidean distance to all \( K \) cluster centers.  
2. Assign the point to the cluster whose center is the closest.  

#### **M-Step (Update Centers)**  
1. For each cluster, calculate the mean of the coordinates of all points assigned to it.  
2. Update the cluster center to this mean.  

### **3. Termination**  
- Stop when the labels of points no longer change between iterations.

---

