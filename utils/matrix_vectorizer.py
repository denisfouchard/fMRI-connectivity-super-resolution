import numpy as np
import torch


class MatrixVectorizer:
    """
    A class for transforming between matrices and vector representations.

    This class provides methods to convert a symmetric matrix into a vector (vectorize)
    and to reconstruct the matrix from its vector form (anti_vectorize), focusing on
    vertical (column-based) traversal and handling of elements.
    """

    def __init__(self):
        """
        Initializes the MatrixVectorizer instance.

        The constructor currently does not perform any actions but is included for
        potential future extensions where initialization parameters might be required.
        """
        pass

    @staticmethod
    def vectorize(matrix, include_diagonal=False):
        """
        Converts a matrix into a vector by vertically extracting elements.

        This method traverses the matrix column by column, collecting elements from the
        upper triangle, and optionally includes the diagonal elements immediately below
        the main diagonal based on the include_diagonal flag.

        Parameters:
        - matrix (numpy.ndarray): The matrix to be vectorized.
        - include_diagonal (bool, optional): Flag to include diagonal elements in the vectorization.
          Defaults to False.

        Returns:
        - numpy.ndarray: The vectorized form of the matrix.
        """
        # Determine the size of the matrix based on its first dimension
        matrix_size = matrix.shape[0]

        # Initialize an empty list to accumulate vector elements
        vector_elements = []

        # Iterate over columns and then rows to collect the relevant elements
        for col in range(matrix_size):
            for row in range(matrix_size):
                # Skip diagonal elements if not including them
                if row != col:
                    if row < col:
                        # Collect upper triangle elements
                        vector_elements.append(matrix[row, col])
                    elif include_diagonal and row == col + 1:
                        # Optionally include the diagonal elements immediately below the diagonal
                        vector_elements.append(matrix[row, col])

        return np.array(vector_elements)

    @staticmethod
    def anti_vectorize(vector, matrix_size, include_diagonal=False):
        """
        Efficiently reconstructs a symmetric matrix from its vector form.

        Parameters:
        - vector (torch.Tensor): The vector to be transformed into a matrix.
        - matrix_size (int): The size of the square matrix to be reconstructed.
        - include_diagonal (bool, optional): Flag to include diagonal elements. Defaults to False.

        Returns:
        - torch.Tensor: The reconstructed symmetric square matrix.
        """
        # Create symmetric indices for upper triangle
        indices = torch.triu_indices(matrix_size, matrix_size, offset=1)

        # Initialize the matrix
        matrix = torch.zeros((matrix_size, matrix_size), device=vector.device
        if hasattr(vector, 'device') else None)

        # Number of elements in the upper triangle (excluding diagonal)
        n_elements = indices.shape[1]

        # Ensure the vector has the right number of elements
        assert vector.shape[0] >= n_elements, "Vector size doesn't match required elements"

        # Fill upper and lower triangles simultaneously
        matrix[indices[0], indices[1]] = vector[:n_elements]
        matrix[indices[1], indices[0]] = vector[:n_elements]

        # Handle diagonal if needed
        if include_diagonal:
            diag_idx = torch.arange(matrix_size)
            matrix[diag_idx, diag_idx] = vector[n_elements:n_elements + matrix_size]

        return matrix
