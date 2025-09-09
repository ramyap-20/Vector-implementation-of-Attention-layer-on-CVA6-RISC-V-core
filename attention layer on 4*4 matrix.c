#include <stdio.h> 
#include <math.h> 
 
#define SIZE 4 
#define SQRT_DK 2.0  // sqrt(4) since d_k = 4 

void print_matrix(const char* name, float matrix[SIZE][SIZE]) { 
    printf("%s:\n", name); 
    for (int i = 0; i < SIZE; i++) { 
        for (int j = 0; j < SIZE; j++) { 
            printf("%8.4f ", matrix[i][j]); 
        } 
        printf("\n"); 
    } 
    printf("\n"); 
} 
 
void softmax_row(float input[SIZE], float output[SIZE]) { 
    float max = input[0]; 
    for (int i = 1; i < SIZE; i++) { 
        if (input[i] > max) 
            max = input[i]; 
    } 
 
    float sum = 0.0; 
    for (int i = 0; i < SIZE; i++) { 
        output[i] = exp(input[i] - max);  // For numerical stability 
        sum += output[i]; 
    } 
 
    for (int i = 0; i < SIZE; i++) { 
        output[i] /= sum; 
    } 
} 
 
int main() { 
    
    float Q[SIZE][SIZE] = { 
        {1, 0, 1, 0}, 
        {0, 1, 0, 1}, 
        {1, 1, 0, 0}, 
        {0, 0, 1, 1} 
    }; 
 
    float K[SIZE][SIZE] = { 
        {1, 0, 1, 0}, 
        {0, 1, 0, 1}, 
        {1, 1, 0, 0}, 
        {0, 0, 1, 1} 
    }; 
 
    float V[SIZE][SIZE] = { 
        {0.1, 0.2, 0.3, 0.4}, 
        {0.5, 0.6, 0.7, 0.8}, 
        {0.9, 1.0, 1.1, 1.2}, 
        {1.3, 1.4, 1.5, 1.6} 
    }; 
 
    float scores[SIZE][SIZE] = {0}; 
    float scaled[SIZE][SIZE] = {0}; 
    float weights[SIZE][SIZE] = {0}; 
    float output[SIZE][SIZE] = {0}; 
 
    
    for (int i = 0; i < SIZE; i++) { 
        for (int j = 0; j < SIZE; j++) { 
            float dot = 0.0; 
            for (int k = 0; k < SIZE; k++) { 
                dot += Q[i][k] * K[j][k];  // Note: K[j] because we're doing K^T 
            } 
            scores[i][j] = dot; 
        } 
    } 
 
    print_matrix("1. Scores (Q x Kᵀ)", scores); 
 

    for (int i = 0; i < SIZE; i++) 
        for (int j = 0; j < SIZE; j++) 
            scaled[i][j] = scores[i][j] / SQRT_DK; 
 
    print_matrix("2. Scaled Scores", scaled); 
 
   
    for (int i = 0; i < SIZE; i++) 
        softmax_row(scaled[i], weights[i]); 
 
    print_matrix("3. Attention Weights (Softmax)", weights); 
 
     
    for (int i = 0; i < SIZE; i++) { 
        for (int j = 0; j < SIZE; j++) { 
            float sum = 0.0; 
            for (int k = 0; k < SIZE; k++) { 
                sum += weights[i][k] * V[k][j]; 
            } 
            output[i][j] = sum; 
        } 
    } 
 
    print_matrix("4. Attention Output", output); 
 
    return 0; 
} 
 
 
