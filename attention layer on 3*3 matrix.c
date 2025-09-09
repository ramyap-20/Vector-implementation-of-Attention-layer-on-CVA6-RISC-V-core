#include <stdio.h> 
#include <math.h> 
 
#define SIZE 3 
 
void matrix_multiply(float A[SIZE][SIZE], float B[SIZE][SIZE], float result[SIZE][SIZE]) { 
    for (int i = 0; i < SIZE; i++) { 
        for (int j = 0; j < SIZE; j++) { 
            result[i][j] = 0; 
            for (int k = 0; k < SIZE; k++) { 
                result[i][j] += A[i][k] * B[k][j]; 
            } 
        } 
    } 
} 
 
void softmax(float row[SIZE]) { 
    float sum = 0.0; 
    for (int i = 0; i < SIZE; i++) { 
        row[i] = exp(row[i]); 
        sum += row[i]; 
    } 
    for (int i = 0; i < SIZE; i++) { 
        row[i] /= sum; 
    } 
} 
 
void print_matrix(float matrix[SIZE][SIZE]) { 
    for (int i = 0; i < SIZE; i++) { 
        for (int j = 0; j < SIZE; j++) { 
            printf("%.3f ", matrix[i][j]); 
        } 
        printf("\n"); 
    } 
} 
 
int main() { 
    float Q[SIZE][SIZE] = { 
        {1, 0, 0}, 
        {0, 1, 0}, 
        {0, 0, 1} 
    }; 
     
    float K[SIZE][SIZE] = { 
        {1, 0, 1}, 
        {0, 1, 0}, 
        {1, 0, 1} 
    }; 
 
    float V[SIZE][SIZE] = { 
        {0, 1, 1}, 
        {1, 0, 0}, 
        {1, 1, 0} 
    }; 
     
    float QK_T[SIZE][SIZE]; 
    matrix_multiply(Q, K, QK_T);  // Q * K^T 
 
 
    float scaling_factor = sqrt((float)SIZE); 
    for (int i = 0; i < SIZE; i++) { 
        for (int j = 0; j < SIZE; j++) { 
            QK_T[i][j] /= scaling_factor; 
        } 
    } 
 
    // Apply softmax to each row of QK_T 
    for (int i = 0; i < SIZE; i++) { 
        softmax(QK_T[i]); 
    } 
 
    printf("Softmax Output:\n"); 
    print_matrix(QK_T); 
 
    
    float attention_output[SIZE][SIZE]; 
    matrix_multiply(QK_T, V, attention_output); 
 
    printf("\nAttention Output:\n"); 
    print_matrix(attention_output); 
 
    return 0; 
} 
 
 
 
