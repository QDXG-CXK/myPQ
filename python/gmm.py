'''
generate queries
'''

import os
import re
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


PATH_QUERY = "/data/sift10m/query.bin"
n_samples = 1000

def read_bin(fname):
    # Parse the postfix of the file name with Regular Expression
    match = re.search(r'(\D+)(\d*)bin$', fname.split('.')[-1])
    if not match:
        raise ValueError("Invalid postfix, must end with *bin")
    data_type_prefix = match.group(1)
    bits = match.group(2)

    if bits == '':
        bits = '32'
    bits = int(bits)
    
    if data_type_prefix == 'i':
        dtype = np.dtype(f'int{bits}')
    elif data_type_prefix == 'u':
        dtype = np.dtype(f'uint{bits}')
    elif data_type_prefix == 'f':
        dtype = np.dtype(f'float{bits}')
    else:
        raise ValueError("Invalid data type, must be 'i', 'u', or 'f'")
    
    shape = np.fromfile(fname, dtype=np.uint32, count=2)
    data = np.memmap(fname, dtype=dtype, offset=8, mode='r').reshape(shape)
    return data, shape[0], shape[1]

def get_mahalanobis_distance(base_matrix, test_matrix, inv_cov_matrix = None):
    """
    Compute the Mahalanobis distance between a base matrix and a test matrix.

    Parameters:
    base_matrix (np.array): The base matrix.
    test_matrix (np.array): The test matrix.

    Returns:
    distance (float): The Mahalanobis distance.
    """
    if inv_cov_matrix is None:
        # Compute the covariance matrix of the base matrix
        cov_matrix = np.cov(base_matrix.T)

        # Compute the inverse of the covariance matrix
        inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Compute the mean of the base matrix
    base_mean = np.mean(base_matrix, axis=0)

    # Compute the difference between the test matrix and the mean matrix
    diff_matrix = test_matrix - base_mean

   
    # Compute the Mahalanobis distance
    left_part = np.dot(diff_matrix, inv_cov_matrix)
    distance = np.einsum('ij,ji->i', left_part, diff_matrix.T)

    # distance = np.dot(np.dot(diff_matrix, inv_cov_matrix), diff_matrix.T)

    # return np.diag(distance)
    return distance

def plot_mahalanobis_distance_distribution(X, Q, dataset):
    SQ = np.random.choice(Q.shape[0], 500, replace=False)
    ma_dist = get_mahalanobis_distance(X, X[SQ])
    print(f'query ma_dist min: {np.min(ma_dist)}, max: {np.max(ma_dist)}')
    # randomly sample 10,000 base vectors
    S = np.random.choice(X.shape[0], 500, replace=False)
    base_ma_dist = get_mahalanobis_distance(X, X[S])
    print(f'base ma_dist min: {np.min(base_ma_dist)}, max: {np.max(base_ma_dist)}')
    
    plt.hist(base_ma_dist, bins=50, edgecolor='black', label='base', color='orange')
    plt.hist(ma_dist, bins=50, edgecolor='black', label='query', color='steelblue')
    plt.xlabel('Mahalanobis distance')
    plt.ylabel('number of points')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.title(f'{dataset} Mahalanobis distance distribution')
    plt.show()
    # plt.savefig(f'./figures/{dataset}/{dataset}-mahalanobis-distance-distribution.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-mahalanobis-distance-distribution.png')

if __name__ == '__main__':
    # x_query, n_query, dim_query = read_bin(PATH_QUERY)
    x_query = np.memmap(PATH_QUERY, dtype=np.float32, offset=0, mode='r').reshape((1000, 128)) # debug
    gmm = GaussianMixture(n_components=4, covariance_type='full', verbose=True).fit(x_query)
    print('model built')
    X_sample = gmm.sample(n_samples)[0]

    plot_mahalanobis_distance_distribution(x_query, X_sample, x_query)
        
        
        
        
