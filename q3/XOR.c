#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <fcntl.h>

#define PATTERNS  4
#define NUMIN     2
#define NUMOUT    1
#define NUMHIDDEN 2

#define EPOCH     20000
#define ETA       0.6
#define ALPHA     0.9
#define MAXERROR  0.001

int main() {
    // DECLARATION OF VARIABLES
    int i, j, k;
    int p;
    
    double weight[NUMIN+NUMOUT+NUMHIDDEN][NUMIN+NUMOUT+NUMHIDDEN]       =  {{0.0, 0.0, 0.0, 0.0, 0.0},
                                                                            {0.0, 0.0, 0.0, 0.0, 0.0},
                                                                            {1.0, 1.0, 1.0, 0.0, 0.0},
                                                                            {1.0, 1.0, 1.0, 0.0, 0.0},
                                                                            {1.0, 0.0, 0.0, 1.0, 1.0}}; 
    double delta_weight[NUMIN+NUMOUT+NUMHIDDEN][NUMIN+NUMOUT+NUMHIDDEN] =  {{0.0, 0.0, 0.0, 0.0, 0.0},
                                                                            {0.0, 0.0, 0.0, 0.0, 0.0},
                                                                            {0.0, 0.0, 0.0, 0.0, 0.0},
                                                                            {0.0, 0.0, 0.0, 0.0, 0.0},                                                                            
                                                                            {0.0, 0.0, 0.0, 0.0, 0.0}};
    double error_gradient[PATTERNS][NUMIN+NUMOUT+NUMHIDDEN]             =   {{0.0}, {0.0}, {0.0}, {0.0}};
    int    input[PATTERNS][NUMIN]                                       =  {{0, 0},
                                                                            {0, 1},
                                                                            {1, 0},
                                                                            {1, 1}};
    double target[PATTERNS][NUMOUT]                                     =  { {0.0},
                                                                             {1.0},
                                                                             {1.0},
                                                                             {0.0}};
    double node_output[NUMIN+NUMOUT+NUMHIDDEN]                          =  { 0.0, 
                                                                             0.0, 
                                                                             0.0, 
                                                                             0.0};
    double error                                                        =    0.0;
    
    int np, op;
    int randpattern[PATTERNS];
    
    // SETS WEIGHTS RANDOMLY
       // This will set all the weights randomly.
       
    for (i = 0; i < NUMIN + NUMOUT + NUMHIDDEN; i++) {
       for (j = 0; j < NUMIN + NUMOUT + NUMHIDDEN; j++) {
           if (weight[i][j] != 0) {
               weight[i][j] = 2.0 * (((double)rand()/RAND_MAX) - 0.5) * 0.5;
               printf("weight[%d][%d] %f \n\r", i,j, weight[i][j]);          
              }
            }
        }    
    system("PAUSE");
    
    srand(7);
    
    // THE CODE BODY
    while(1){
         error =    0.0;
         
         // END EXTRA
         for(np = 0; np < PATTERNS; np++){
             // p = randpattern[np];
             p = np;
             
             // UPDATE THE NN.
             // HIDDEN LAYER.
             for(i = 0; i < NUMIN + NUMHIDDEN + NUMOUT ; i++){                
                if(i < NUMIN) {
                     node_output[i] = input[p][i];
                     // printf("i %d %d\n", i, i - (NUMIN + NUMHIDDEN));
                } else {
                     node_output[i] = weight[i][0];
                     for(j = 1; j <= i ; j++) {
                        node_output[i] += weight[i][j] * node_output[j-1];
                     }
                     node_output[i]  = 1.0/(1.0 + exp(-node_output[i]));
                     
                     // Calculates error if it is a output node.
                     if (i > NUMIN + NUMHIDDEN - 1) {
                         error                += 0.5 * (target[p][i - (NUMIN + NUMHIDDEN)] - node_output[i]) * (target[p][i - (NUMIN + NUMHIDDEN)] - node_output[i]);
                         error_gradient[p][i]  = node_output[i] * (1.0 - node_output[i]) * (target[p][i - (NUMIN + NUMHIDDEN)] - node_output[i]);
                     }
                }
             }

             // Finds gradient for the hidden node.
             // Back propagation.
             for(i = NUMIN + NUMHIDDEN - 1 ; i > NUMIN - 1 ; i--){ // Iterates through the hidden nodes.
                error_gradient[p][i] = 0.0;
                for (j = NUMIN + NUMHIDDEN + NUMOUT - 1; j > i; j--) { // Iterates through nodes connected to hidden node.
                    error_gradient[p][i] += weight[j][i+1] * error_gradient[p][j] * node_output[i] * (1.0 - node_output[i]);
                }
             }             
             // End of back propagation.
              
             // Updating hidden nodes and connecting edges.
             for(i=NUMIN; i < NUMIN + NUMHIDDEN + NUMOUT; i++){
                delta_weight[i][0]    = ETA * error_gradient[p][i] + ALPHA * delta_weight[i][0];
                weight[i][0]         += delta_weight[i][0];
                for(j = 1; j < NUMIN + NUMHIDDEN + NUMOUT; j++) {                   
                   if (weight[i][j] != 0){
                       delta_weight[i][j]  = ETA * node_output[j-1] * error_gradient[p][i] + ALPHA * delta_weight[i][j];
                       weight[i][j]       += delta_weight[i][j];
                   }
                }
             }
             //system("PAUSE");
             
             //printf("EG NODE 2 %f %f %f %f\n", error_gradient[0][2], error_gradient[1][2], error_gradient[2][2], error_gradient[3][2]);
             //printf("EG NODE 3 %f %f %f %f\n", error_gradient[0][3], error_gradient[1][3], error_gradient[2][3], error_gradient[3][3]);
             //printf("p %d\n", p);
             printf("input0 %d, input1 %d, output %f, desired %f\n", input[p][0], input[p][1], node_output[4], target[p][0]);
             //printf("node_output[3] %f, node_output[2] %f\n", node_output[3], node_output[2]);
             //printf("%f %f %f %f\n", error_gradient[0][3], error_gradient[1][3], error_gradient[2][3], error_gradient[3][3]);             
             //printf("target[p] -> %f, p -> %d\n", target[p], p);
             }    
    printf("Error -> %f\n\r", error);
    
    if (error < MAXERROR) break;
    }
    
    for (i = 0; i < NUMIN + NUMOUT + NUMHIDDEN; i++) {
       for (j = 0; j < NUMIN + NUMOUT + NUMHIDDEN; j++) {
           if (weight[i][j] != 0) {
               printf("weight[%d][%d] %f, delta_weight[%d][%d] %f\n\r", i,j, weight[i][j], i, j, delta_weight[i][j]);                        
           }
       }
    }    

    system("PAUSE");
    return;    
}

