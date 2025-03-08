import numpy as np
import torch

class MatrixVectorizer:
    """
    A class for transforming between matrices and vector representations.

    This class provides methods to convert a symmetric matrix into a vector (vectorize)
    and to reconstruct the matrix from its vector form (anti_vectorize), focusing on
    vertical (column-based) traversal and handling of elements.
    """

    @staticmethod
    def vectorize(matrix, include_diagonal=False):
        """
        Converts a symmetric matrix into a vector by extracting elements from its upper triangle.

        Parameters:
        - matrix (numpy.ndarray): The symmetric matrix to be vectorized.
        - include_diagonal (bool, optional): Whether to include diagonal elements. Defaults to False.

        Returns:
        - numpy.ndarray: The vectorized form of the matrix.
        """
        # Get upper triangle indices (excluding or including diagonal)
        k = 0 if include_diagonal else 1
        return matrix[np.triu_indices(matrix.shape[0], k=k)]

    @staticmethod
    def anti_vectorize(vector, matrix_size, include_diagonal=False):
        """
        Efficiently reconstructs a symmetric matrix from its vector form.

        Parameters:
        - vector (numpy.ndarray or torch.Tensor): The vector to be transformed into a matrix.
        - matrix_size (int): The size of the square matrix to be reconstructed.
        - include_diagonal (bool, optional): Whether to include diagonal elements. Defaults to False.

        Returns:
        - numpy.ndarray: The reconstructed symmetric square matrix.
        """
        # Initialize a zero matrix
        matrix = np.zeros((matrix_size, matrix_size))

        # Get upper triangle indices
        k = 0 if include_diagonal else 1
        triu_indices = np.triu_indices(matrix_size, k=k)

        # Ensure the vector has the correct number of elements
        assert vector.shape[0] == len(triu_indices[0]), "Vector size doesn't match required elements"

        # Assign values to upper triangle
        matrix[triu_indices] = vector

        # Mirror the values to the lower triangle
        matrix = matrix + matrix.T - np.diag(matrix.diagonal())

        return matrix
