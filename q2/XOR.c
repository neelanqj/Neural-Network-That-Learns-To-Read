#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <fcntl.h>


#define PATTERNS  4
#define NUMIN     2
#define NUMOUT    1
#define NUMHIDDEN 1

#define EPOCH     20000
#define ETA       0.6
#define ALPHA     0.9
#define MAXERROR  0.001

int main() {
    // DECLARATION OF VARIABLES
    int i, j, k;
    int p;
    double weight[NUMIN+NUMOUT+NUMHIDDEN][NUMIN+NUMOUT+NUMHIDDEN]       =  {{0.0, 0.0, 0.0, 0.0},
                                                                            {0.0, 0.0, 0.0, 0.0},
                                                                            {2.2, -6.4, -6.4, 0.0},
                                                                            {6.3, -4.2, -4.2, -9.4}}; 
    double delta_weight[NUMIN+NUMOUT+NUMHIDDEN][NUMIN+NUMOUT+NUMHIDDEN] =  {{0.0, 0.0, 0.0, 0.0},
                                                                            {0.0, 0.0, 0.0, 0.0},
                                                                            {0.0, 0.0, 0.0, 0.0},
                                                                            {0.0, 0.0, 0.0, 0.0}};
    double error_gradient[PATTERNS][NUMIN+NUMOUT+NUMHIDDEN]             =   {0.0, 0.0, 0.0, 0.0};
    int    input[PATTERNS][NUMIN]                                       =  {{0, 0},
                                                                            {0, 1},
                                                                            {1, 0},
                                                                            {1, 1}};
    double target[PATTERNS]                                             =  { 0.0,
                                                                             1.0,
                                                                             1.0,
                                                                             0.0};
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
    
    srand(12);
    
    // THE CODE BODY    
    while(1){
         error =    0.0;        
        
         // END EXTRA         
         for(np=0; np < PATTERNS; np++){
             // p = randpattern[np];
             p = np;
             
             // UPDATE THE NN.
             // HIDDEN LAYER.
             node_output[2]  = weight[2][0];
             node_output[2] += weight[2][0+1] * input[p][0];
             node_output[2] += weight[2][1+1] * input[p][1];
             node_output[2]  = 1.0/(1.0 + exp(-node_output[2]));
             
             // OUTPUT LAYER
             node_output[3]  = weight[3][0];
             node_output[3] += weight[3][0+1] * input[p][0];
             node_output[3] += weight[3][1+1] * input[p][1];
             node_output[3] += weight[3][2+1] * node_output[2];
             node_output[3]  = 1.0/(1.0 + exp(-node_output[3]));
             
             error                += 0.5 * (target[p] - node_output[3]) * (target[p] - node_output[3]);
             error_gradient[p][3]  = node_output[3] * (1.0 - node_output[3]) * (target[p] - node_output[3]);

             // Finds gradient for the hidden node.
             // Back propagation.
             error_gradient[p][2]  = weight[3][2+1] * error_gradient[p][3];
             error_gradient[p][2]  = error_gradient[p][2] * node_output[2] * (1.0 - node_output[2]);
             // End of back propagation.
             
             // Updating hidden nodes and connecting edges
             delta_weight[2][0]    = ETA * error_gradient[p][2] + ALPHA * delta_weight[2][0];
             weight[2][0]         += delta_weight[2][0];
             
             delta_weight[2][0+1]  = ETA * input[p][0] * error_gradient[p][2] + ALPHA * delta_weight[2][0+1];
             weight[2][0+1]       += delta_weight[2][0+1];
             delta_weight[2][1+1]  = ETA * input[p][1] * error_gradient[p][2] + ALPHA * delta_weight[2][1+1];
             weight[2][1+1]       += delta_weight[2][1+1];
             
             // Updating output nodes and connecting edges
             delta_weight[3][0]    = ETA * error_gradient[p][3] + ALPHA * delta_weight[3][0];
             weight[3][0]         += delta_weight[3][0];
             
             delta_weight[3][0+1]  = ETA * input[p][0] * error_gradient[p][3] + ALPHA * delta_weight[3][0+1];
             weight[3][0+1]       += delta_weight[3][0+1];
             delta_weight[3][1+1]  = ETA * input[p][1] * error_gradient[p][3] + ALPHA * delta_weight[3][1+1];
             weight[3][1+1]       += delta_weight[3][1+1];
             delta_weight[3][2+1]  = ETA * node_output[2] * error_gradient[p][3] + ALPHA * delta_weight[3][2+1];
             weight[3][2+1]       += delta_weight[3][2+1];
             
             //printf("EG NODE 2 %f %f %f %f\n", error_gradient[0][2], error_gradient[1][2], error_gradient[2][2], error_gradient[3][2]);
             //printf("EG NODE 3 %f %f %f %f\n", error_gradient[0][3], error_gradient[1][3], error_gradient[2][3], error_gradient[3][3]);
             //printf("p %d\n", p);
             printf("input0 %d, input1 %d, output %f, desired %f\n", input[p][0], input[p][1], node_output[3], target[p]);
             //printf("node_output[3] %f, node_output[2] %f\n", node_output[3], node_output[2]);             
             //printf("%f %f %f %f\n", error_gradient[0][3], error_gradient[1][3], error_gradient[2][3], error_gradient[3][3]);             
             //printf("target[p] -> %f, p -> %d\n", target[p], p);
             }
    
    //printf("Error -> %f\n\r", error);
    
    if (error < MAXERROR) break;
    }
    
    //printf("\n--------------------------\n");
    
    for (i = 0; i < NUMIN + NUMOUT + NUMHIDDEN; i++) {
       for (j = 0; j < NUMIN + NUMOUT + NUMHIDDEN; j++) {
           if (weight[i][j] != 0) {
               printf("weight[%d][%d] %f \n\r", i,j, weight[i][j]);          
              }

            }
        }    

    system("PAUSE");
    return;    
}

