from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from pdc_dp_means import DPMeans
from pdc_dp_means import MiniBatchDPMeans
import matplotlib.pyplot as plt
import time
import numpy as np
import os

pcds_dir = os.path.join("clustering","pcds")
def Plotting_clusters_and_centroids(X, y, clusters, plot_title, met_str):
     #centers = met.cluster_centers_
    # Plotting clusters and centroids
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
    plt.scatter(clusters[:, 0], clusters[:, 1], c='black', s=200, alpha=0.5)    
    plt.title(plot_title)
    plt.savefig(f'clustering/cluster_example_{met_str}.png')
    
def create_blobs(n_samples_blob=190_000, centers_blob=100_000, cluster_std_blob=0.60,  n_features_blob=3):
    X, y_true = make_blobs(n_samples=n_samples_blob, centers=centers_blob, cluster_std=cluster_std_blob, random_state=0, n_features=n_features_blob)
    return X, y_true

def cluster(X, y=None, n_clusters_met=0,
            str_met="dpmeans",
            delta=5, max_clusters=1000,
             save_clusters_and_centroids_flag=False):   
    # Record the start time
    start_time = time.time()

    if n_clusters_met == 0:
        n_clusters_met = X.shape[0]//2

    if X.shape[0] < n_clusters_met:
        n_clusters_met = X.shape[0]
        
    # Apply clustering
    
    if str_met == "kmeans":
        max_clusters = n_clusters_met
        met = KMeans(n_clusters = n_clusters_met, random_state = 0, n_init='auto')
        met.fit(X)
    elif str_met == "kmeansMiniBatch":
        met = MiniBatchKMeans(n_clusters = n_clusters_met, random_state = 0, n_init='auto') # best performance batch_size=256*cpu_cores
        met.fit(X)
    elif str_met == "dpmeansMiniBatch":
        met = MiniBatchDPMeans(n_clusters = n_clusters_met, random_state = 0, n_init=1, delta=delta) # best performance batch_size=256*cpu_cores
        met.fit(X)
    elif str_met == "dpmeans":
        met = DPMeans(n_clusters = n_clusters_met, random_state = 0, n_init=1, delta=delta, max_clusters=max_clusters)
        met.fit(X)
    
    if save_clusters_and_centroids_flag:
        save_clusters_and_centroids(X, met, str_met=str_met)
    
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    def seconds_to_minutes_and_seconds(seconds):
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return minutes, remaining_seconds
    minutes, seconds = seconds_to_minutes_and_seconds(elapsed_time)
    title = f'Clustering {str_met} (request n_clusters_met={n_clusters_met}, max_clusters={max_clusters}, points={X.shape[0]}, delta={delta}). Final:{met.cluster_centers_.shape} Clusters\n(Time: {minutes:.2f} minutes and {seconds:.2f} seconds)'
    # Record the end time
    return title

def save_clusters_and_centroids(X, met, str_met="kmeans"):
    # Save clusters and centroids
    preds = met.predict(X)
    clusters = met.cluster_centers_
    num_of_clusters = clusters.shape[0]
    np.save(os.path.join(pcds_dir, str_met, f"train_preds_{num_of_clusters}.npy"), preds)
    np.save(os.path.join(pcds_dir, str_met, f"train_clusters_{num_of_clusters}.npy"), clusters)
    
    # Plotting clusters and centroids
def write_to_file(t, file_path):
    with open(file_path, "a") as f:
        f.write(f"{t}\n\n") 

if __name__ == "__main__":
    with open(os.path.join(pcds_dir,"cluster_example.txt"), "w") as f:
        f.write(f"My Tests:\n\n") 
    X_points = np.load(os.path.join(pcds_dir,"train_points.npy"))
    #X_points = X_points[:2000, :]
    file_path = os.path.join(pcds_dir,"cluster_example.txt")
    
    
    # t = cluster(str_met="kmeans", X=X_points, n_clusters_met=100)
    # write_to_file(t, file_path)    
    # t = cluster(str_met="kmeans", X=X_points)
    # write_to_file(t, file_path)
    t = cluster(str_met="kmeansMiniBatch", X=X_points, n_clusters_met=50_000, save_clusters_and_centroids_flag=True, delta=0.05)
    write_to_file(t, file_path)
    
    t = cluster(str_met="dpmeansMiniBatch", X=X_points, n_clusters_met=50_000, save_clusters_and_centroids_flag=True, delta=0.05)
    write_to_file(t, file_path)
    
    # t = cluster(str_met="dpmeans", X=X_points, n_clusters_met=100, max_clusters=2000, delta=0.05)
    # write_to_file(t, file_path)
    # t = cluster(str_met="dpmeans", X=X_points, n_clusters_met=90_000, max_clusters=100_000, delta=0.05)
    # write_to_file(t, file_path)
    # t = cluster(str_met="dpmeans", X=X_points, n_clusters_met=150_000, max_clusters=180_000, delta=0.05)
    # write_to_file(t, file_path)
    
    # t = cluster(str_met="dpmeans", X=X_points, n_clusters_met=50_500, max_clusters=150_000, delta=0.001, save_clusters_and_centroids_flag=True)
    # write_to_file(t, file_path)
    
