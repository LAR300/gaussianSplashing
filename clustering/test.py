import numpy as np
import os
pcds_dir = os.path.join("clustering","pcds")
if __name__ == "__main__":
    # Example arrays A and B
    A = np.load(os.path.join(pcds_dir,"train_points.npy"))
    print(A.shape)
    B = np.reshape(np.load(os.path.join(pcds_dir,"kmeans", "train_preds_50000.npy")), (-1, 1))   
    print(B.shape)
    # Combine A and B into a single array C
    C = np.concatenate((A, B), axis=1)

    # Use np.unique to get unique values in the last column (B)
    unique_values = np.unique(C[:, -1])

    # Create an empty array R to store the result
    R = np.zeros((len(unique_values), A.shape[1]))

    # Iterate over unique values in B and calculate the average for each group
    for i, value in enumerate(unique_values):
        # Select rows where the last column is equal to the current unique value
        group = A[B.flatten() == value]
        # Calculate the average for each column in the group
        average_point = np.mean(group, axis=0)
        # Assign the average to the corresponding row in R
        R[i, :] = average_point

    print("Original A:")
    print(A[:10,:])

    print("\nOriginal B:")
    print(B[:5,:])

    print("\nResult R:")
    print(R[:10,:])
    print(R.shape)