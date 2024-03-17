/* Copyright Dr. Xiang Fu
    Author: Trevor Conley
    All Rights Reserved.
    Created: 06/30/2023
    Provides functions for generating a toeplitx matrix from a vector.
    Provides functions for multiplying 2 matrices given a vector.
*/
/* vec = generate_toeplitz(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
	[1.0, 2.0, 3.0, 4.0, 5.0]
	[2.0, 1.0, 2.0, 3.0, 4.0]
	[3.0, 2.0, 1.0, 2.0, 3.0]
	[4.0, 3.0, 2.0, 1.0, 2.0]
	[5.0, 4.0, 3.0, 2.0, 1.0]*/
/* This function generates a Toeplitz matrix given a vector. A Toeplitz matrix is a diagonal-constant matrix,
   i.e., all the elements on a given diagonal are the same. The input vector provides the first row of the matrix,
   and each subsequent row is a shifted version of the previous one, resulting in the diagonal-constant property.
   This function assumes that the input vector provides the main diagonal elements, with the remaining elements 
   defined such that each diagonal of the matrix holds the same values.*/
pub fn generate_toeplitz(vector: Vec<f64>) -> Vec<Vec<f64>> {

	// Define the size of the matrix
    let n = vector.len();
    
    // Initialize the matrix with zeros
    let mut matrix = vec![vec![0.0; n]; n];
    
    // Iterate over each element in the matrix
    for i in 0..n {
        for j in 0..n {
            // If i > j, use the difference as the index into the vector for the value
            if i > j {
                matrix[i][j] = vector[i - j];
            } 
            // If i <= j, use the difference as the index into the vector for the value
            else {
                matrix[i][j] = vector[j - i];
            }
        }
    }
    
    // Return the generated Toeplitz matrix
    matrix

}
/* This function takes two vectors as input, generates Toeplitz matrices from them,
and then performs an element-wise multiplication (Hadamard product) of the two matrices.
The result is a new matrix where each element is the product of the corresponding elements
in the two input matrices. The input vectors must be of the same length.*/
pub fn toeplitz_mult(vector1: Vec<f64>, vector2: Vec<f64>) -> Vec<Vec<f64>> {
    
	// Assert that the two input vectors have the same length
    assert_eq!(
        vector1.len(),
        vector2.len(),
        "Vectors must be of the same length"
    );

    // Generate Toeplitz matrices from the input vectors
    let matrix1 = generate_toeplitz(vector1);
    let matrix2 = generate_toeplitz(vector2);

    // Get the size of the matrices
    let n = matrix1.len();

    // Initialize the result matrix with zeros
    let mut result = vec![vec![0.0; n]; n];

    // Perform the element-wise multiplication of the two matrices
    for i in 0..n {
        for j in 0..n {
            result[i][j] = matrix1[i][j] * matrix2[i][j];
        }
    }

    // Return the resultant matrix after multiplication
    result

}

#[cfg(test)]
mod tests {
    use super::*;
    use izpr::serial_poly_utils::ark_ff::{PrimeField, Zero};
    use izpr::serial_poly_utils::ark_poly::{univariate::DensePolynomial, Polynomial};
    use izpr::serial_poly_utils::ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use izpr::serial_poly_utils::ark_std::{log2, rand::Rng};
    use izpr::serial_poly_utils::{get_time, rand_arr_field_ele, vec_equals};

    #[test]
    fn test_topelitz() {
        let vector1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vector2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let matrix1 = generate_toeplitz(vector1.clone());
        let matrix2 = generate_toeplitz(vector2.clone());
        let result = toeplitz_mult(vector1, vector2);

        // Define the expected result of the multiplication
        let expected_result = vec![
            vec![1.0, 4.0, 9.0, 16.0, 25.0],
            vec![4.0, 1.0, 4.0, 9.0, 16.0],
            vec![9.0, 4.0, 1.0, 4.0, 9.0],
            vec![16.0, 9.0, 4.0, 1.0, 4.0],
            vec![25.0, 16.0, 9.0, 4.0, 1.0],
        ];

        // Assert that the actual result is equal to the expected result
        assert_eq!(result, expected_result, "Toeplitz multiplication failed");
    }
}

