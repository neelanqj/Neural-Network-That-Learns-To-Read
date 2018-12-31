#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <fcntl.h>
#define TEST_PATTERNS 5
#define PATTERNS  24
#define NUMIN     64
#define NUMOUT    5
#define NUMHIDDEN 61

#define ETA       0.5
#define ALPHA     0.3
#define MAXERROR  0.001
#define SEED      10

int main() {
    // DECLARATION OF VARIABLES
    int i, j, k;
    int p;
    
    double weight[NUMIN+NUMOUT+NUMHIDDEN][NUMIN+NUMOUT+NUMHIDDEN];
    double delta_weight[NUMIN+NUMOUT+NUMHIDDEN][NUMIN+NUMOUT+NUMHIDDEN] =  {{0.0, 0.0, 0.0, 0.0, 0.0},
                                                                            {0.0, 0.0, 0.0, 0.0, 0.0},
                                                                            {0.0, 0.0, 0.0, 0.0, 0.0},
                                                                            {0.0, 0.0, 0.0, 0.0, 0.0},
                                                                            {0.0, 0.0, 0.0, 0.0, 0.0}};
    double error_gradient[PATTERNS][NUMIN+NUMOUT+NUMHIDDEN];

    int    input[PATTERNS][NUMIN];
                                                                            
    double target[PATTERNS][NUMOUT]; 

    double test_input[TEST_PATTERNS][NUMIN];

    double node_output[NUMIN+NUMOUT+NUMHIDDEN];
    double error                                                        =    0.0;
    int epoch                                                           =    0.0;
    int np, op;
    int randpattern[PATTERNS];
    int temp;

    FILE   *training_data, *training_expected_results, *test_data;    
    training_data             = fopen("training.txt", "r");
    test_data                 = fopen("test.txt", "r");    
    training_expected_results = fopen("training_expected_results.txt", "r");
    
    // The folowing gets the input from an input file.
    if(!training_data) {
      printf("<ERROR>\n");
      system("PAUSE");
      exit(1);
    }
    for(j=0; j < PATTERNS; j++) {
       for (i=0; i < NUMIN; i++) {
           while((temp = fgetc(training_data)) != EOF) {
               input[j][i] = temp;        
               if (input[j][i] == 10) printf("\n");
               if(input[j][i] == 48 || input[j][i] == 49) break;
           }
       
       input[j][i] = input[j][i] - 48;
       printf("%d", input[j][i]);
       }
    }
    printf("\nFinished loading input training patterns\n");
    system("PAUSE");
//---
    for(j=0; j < PATTERNS; j++) {
       for (i=0; i < NUMOUT; i++) {
           while((temp = fgetc(training_expected_results)) != EOF) {
               target[j][i] = temp;
               if (target[j][i] == 10) printf("\n");
               if(target[j][i] == 48 || target[j][i] == 49) break;
           }
       target[j][i] = target[j][i] - 48;
       printf("%1.0f", target[j][i]);
       }
    }
    printf("\nNote: the colums represent the different output nodes, and the rows the outputs for each of the inputs.\n");
    system("PAUSE");
//---

    for(j=0; j < TEST_PATTERNS; j++) {
       for (i=0; i < NUMIN; i++) {
           while((temp = fgetc(test_data)) != EOF) {
               test_input[j][i] = temp;
               if (test_input[j][i] == 10) printf("\n");
               if(test_input[j][i] == 48 || test_input[j][i] == 49) break;
           }       
       test_input[j][i] = test_input[j][i] - 48;
       printf("%1.0f", test_input[j][i]);
       }
    }
    printf("\nFinished loading training pattern expected results data\n");
    system("PAUSE");

//---

    

// --       
    // Hooks up all hidden node
    printf("Setting node connections");
    for (i = NUMIN; i < NUMIN + NUMOUT + NUMHIDDEN; i++) {
        for (j = 0; j < NUMIN + NUMOUT + NUMHIDDEN; j++) {
            if (j <= NUMIN && i < NUMIN + NUMHIDDEN) {
               weight[i][j] = 1; // Connect input nodes to hidden
            } else if (j > NUMIN && j <= NUMIN+NUMHIDDEN && i >=NUMIN+NUMHIDDEN) {
               weight[i][j] = 1; // Connect input nodes to hidden
            } else {
               break;
            }
        }
    }
    
    srand(SEED);    
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
    printf("\nFinished loading random initial weights\n");
    system("PAUSE");
            
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
                     node_output[i]  = 1.0/(1.0 + exp(-2*node_output[i]));
                     
                     // Calculates error if it is a output node.
                     if (i > NUMIN + NUMHIDDEN - 1) {
                         error                += 0.5 * (target[p][i - (NUMIN + NUMHIDDEN)] - node_output[i]) * (target[p][i - (NUMIN + NUMHIDDEN)] - node_output[i]);
                         error_gradient[p][i]  = 2*node_output[i] * (1.0 - node_output[i]) * (target[p][i - (NUMIN + NUMHIDDEN)] - node_output[i]);
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
             //printf("input0 %d, input1 %d, output %f, desired %f\n", input[p][0], input[p][1], node_output[4], target[p][0]);
             //printf("node_output[3] %f, node_output[2] %f\n", node_output[3], node_output[2]);
             //printf("%f %f %f %f\n", error_gradient[0][3], error_gradient[1][3], error_gradient[2][3], error_gradient[3][3]);             
             //printf("target[p] -> %f, p -> %d\n", target[p], p);
             epoch++;
             }    
    // if ((epoch % 1000) == 1) printf("Error -> %f\n\r", error);
    // printf("Error -> %f\n\r", error);
    if (error < MAXERROR) break;
    }
// END EXTRA

    printf("\nDISPLAYING TEST RESULTS\n");
//Test case
    for (k = 0; k < 5; k++) {
         for(i = 0; i < NUMIN + NUMHIDDEN + NUMOUT ; i++){
            if(i < NUMIN) {
                 node_output[i] = test_input[k][i];
                 // printf("i %d %d\n", i, i - (NUMIN + NUMHIDDEN));
            } else {
                 node_output[i] = weight[i][0];
                 for(j = 1; j <= i ; j++) {
                    node_output[i] += weight[i][j] * node_output[j-1];
                 }
                 node_output[i]  = 1.0/(1.0 + exp(-2*node_output[i]));
                 
                 // Calculates error if it is a output node.
                 if (i > NUMIN + NUMHIDDEN - 1) {
                     error                += 0.5 * (target[p][i - (NUMIN + NUMHIDDEN)] - node_output[i]) * (target[p][i - (NUMIN + NUMHIDDEN)] - node_output[i]);
                     error_gradient[p][i]  = 2*node_output[i] * (1.0 - node_output[i]) * (target[p][i - (NUMIN + NUMHIDDEN)] - node_output[i]);
                 }
            }
         }
         printf("output rounded (%d) > %1.1f, %1.1f, %1.1f, %1.1f, %1.1f\n", k+1, node_output[NUMIN+NUMHIDDEN], node_output[NUMIN+NUMHIDDEN+1], node_output[NUMIN+NUMHIDDEN+2], node_output[NUMIN+NUMHIDDEN+3], node_output[NUMIN+NUMHIDDEN+4]);
    }
    
    system("PAUSE");
    
    for (i = 0; i < NUMIN + NUMOUT + NUMHIDDEN; i++) {
       for (j = 0; j < NUMIN + NUMOUT + NUMHIDDEN; j++) {
           if (weight[i][j] != 0) {
               printf("weight[%d][%d] %f, delta_weight[%d][%d] %f\n\r", i,j, weight[i][j], i, j, delta_weight[i][j]);
           }
       }
    }

    printf("epoch %d\n", epoch);
    system("PAUSE");
    return;
}

