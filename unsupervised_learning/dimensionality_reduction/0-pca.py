import numpy as np

def pca(X, var=0.95):
    """
    Performs PCA on a dataset to reduce dimensionality while maintaining a specified variance.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d), where n is the number of data points 
                           and d is the number of dimensions.
        var (float): Fraction of the variance to maintain after the PCA transformation.

    Returns:
        numpy.ndarray: The weight matrix W of shape (d, nd), where nd is the new dimensionality.
    """
    # Step 1: Compute the covariance matrix
    cov_matrix = np.cov(X, rowvar=False)
    
    # Step 2: Get eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Step 3: Sort eigenvalues and their corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Step 4: Compute the cumulative sum of the eigenvalues to find the required number of components
    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    
    # Step 5: Determine the number of components to maintain the specified variance
    num_components = np.searchsorted(cumulative_variance, var) + 1
    
    # Step 6: Extract the first 'num_components' eigenvectors to form the weights matrix
    W = eigenvectors[:, :num_components]
    
    return W
