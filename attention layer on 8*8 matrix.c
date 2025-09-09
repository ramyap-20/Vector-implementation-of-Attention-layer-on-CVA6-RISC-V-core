#include <stdio.h> 
#include <math.h> 
 
#define SIZE 8 
#define SQRT_DK 2.82842712f  // sqrt(8) 
 
void print_matrix(const char* title, float matrix[SIZE][SIZE]) { 
    printf("%s:\n", title); 
    for (int i = 0; i < SIZE; ++i) { 
        for (int j = 0; j < SIZE; ++j) 
            printf("%8.4f ", matrix[i][j]); 
        printf("\n"); 
    } 
    printf("\n"); 
} 
 
void softmax(float input[SIZE], float output[SIZE]) { 
    float max = input[0]; 
    for (int i = 1; i < SIZE; ++i) 
        if (input[i] > max) max = input[i]; 
 
    float sum = 0.0f; 
    for (int i = 0; i < SIZE; ++i) { 
        output[i] = expf(input[i] - max); 
        sum += output[i]; 
    } 
    for (int i = 0; i < SIZE; ++i) 
        output[i] /= sum; 
} 
 
int main() { 
  
    float Q[SIZE][SIZE] = { 
        {1,0,1,0,1,0,1,0}, 
        {0,1,0,1,0,1,0,1}, 
        {1,1,1,1,0,0,0,0}, 
        {0,0,0,0,1,1,1,1}, 
        {1,0,0,1,1,0,0,1}, 
        {0,1,1,0,0,1,1,0}, 
        {1,1,0,0,1,1,0,0}, 
        {0,0,1,1,0,0,1,1} 
    }; 
 
    float K[SIZE][SIZE]; 
    float V[SIZE][SIZE] = { 
        {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, 
        {0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6}, 
        {1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4}, 
        {2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2}, 
        {3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0}, 
        {4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8}, 
        {4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6}, 
        {5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4} 
    }; 
 
  
    for (int i = 0; i < SIZE; ++i) 
        for (int j = 0; j < SIZE; ++j) 
            K[i][j] = Q[i][j]; 
 
    float scores[SIZE][SIZE] = {0}; 
    float scaled[SIZE][SIZE] = {0}; 
    float weights[SIZE][SIZE] = {0}; 
    float output[SIZE][SIZE] = {0}; 
 
  
    for (int i = 0; i < SIZE; ++i) { 
        for (int j = 0; j < SIZE; ++j) { 
            float dot = 0.0f; 
            for (int k = 0; k < SIZE; ++k) 
                dot += Q[i][k] * K[j][k]; 
            scores[i][j] = dot; 
        } 
    } 
 
    
    for (int i = 0; i < SIZE; ++i) 
        for (int j = 0; j < SIZE; ++j) 
            scaled[i][j] = scores[i][j] / SQRT_DK; 
 
    
    for (int i = 0; i < SIZE; ++i) 
        softmax(scaled[i], weights[i]); 
 
    
    for (int i = 0; i < SIZE; ++i) { 
        for (int j = 0; j < SIZE; ++j) { 
            float sum = 0.0f; 
            for (int k = 0; k < SIZE; ++k) 
                sum += weights[i][k] * V[k][j]; 
            output[i][j] = sum; 
        } 
    } 
 

    print_matrix("Scores (Q x Kᵀ)", scores); 
    print_matrix("Scaled Scores", scaled); 
    print_matrix("Attention Weights (Softmax)", weights); 
    print_matrix("Final Attention Output", output); 
 
    return 0; 
} 
