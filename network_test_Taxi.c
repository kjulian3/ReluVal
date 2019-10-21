/*
 ------------------------------------------------------------------
 ** Top contributors:
 **   Shiqi Wang and Suman Jana
 ** This file is part of the ReluVal project.
 ** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
 ** and their institutional affiliations.
 ** All rights reserved.
 -----------------------------------------------------------------
 *
 * This is the main file of ReluVal, here is the usage:
 * ./network_test [network]
 *      [need to print=0] [test for one run=0] [check mode=0]
 *
 * [network]: the network want to test with
 *
 * [need to print]: whether need to print the detailed info of each split.
 * 0 is not and 1 is yes. Default value is 0.
 *
 * [test for one run]: whether need to estimate the output range without
 * split. 0 is no, 1 is yes. Default value is 0.
 *
 * [check mode]: normal split mode is 0. Check adv mode is 1.
 * Check adv mode will prevent further splits as long as the depth goes
 * upper than 20 so as to locate the concrete adversarial examples faster.
 * Default value is 0.
 * 
 */


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "split.h"
#include <math.h>

//extern int thread_tot_cnt;

/* print the progress if getting SIGQUIT */
void sig_handler(int signo)
{

    if (signo == SIGQUIT) {
        printf("progress: %d/1024\n", progress);
    }

}



int main( int argc, char *argv[])
{
    char *FULL_NET_PATH;

    PROPERTY=1;
    int target=0;
    float delta = 0.0149;

    if (argc > 9 || argc < 2) {
        printf("please specify a network\n");
        printf("./network_test [network] [delta]"
            "[print] "
            "[test for one run] [check mode]\n");
        exit(1);
    }

    for (int i=1;i<argc;i++) {

        if (i == 1) {
            FULL_NET_PATH = argv[i];
        }
        
        if (i == 2) {
            delta = atof(argv[i]);
        }

        if (i == 3) {
            NEED_PRINT = atoi(argv[i]);
            if(NEED_PRINT != 0 && NEED_PRINT!=1){
                printf("Wrong print");
                exit(1);
            }
        }

        if (i == 4) {
            NEED_FOR_ONE_RUN = atoi(argv[i]);

            if (NEED_FOR_ONE_RUN != 0 && NEED_FOR_ONE_RUN != 1) {
                printf("Wrong test for one run");
                exit(1);
            }

        }

        if (i == 5) {

            if (atoi(argv[i]) == 0) {
                CHECK_ADV_MODE = 0;
                PARTIAL_MODE = 0;
            }

            if (atoi(argv[i]) == 1) {
                CHECK_ADV_MODE = 1;
                PARTIAL_MODE = 0;
            }

            if (atoi(argv[i]) == 2) {
                CHECK_ADV_MODE = 0;
                PARTIAL_MODE = 1;
            }

        }

    }

    openblas_set_num_threads(1);
    
    //clock_t start, end;
    srand((unsigned)time(NULL));
    double time_spent;
    int i,j,layer;

    struct NNet* nnet = load_network(FULL_NET_PATH, target);  
    
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;
    
    // Construct the input upper/lower bounds
    // The input bounds are centered around an 128-pixel grayscale image with a maximum
    // pixel-wise disturbance of delta (given as an input).
    float u[inputSize], l[inputSize];
    float img[] = {0.54420954, 0.5643076 , 0.5498468 , 0.52876836, 0.4885723 , 0.43685663, 0.49420956, 0.6616115 , 0.68563116, 0.51651347,       0.51945466, 0.5155331 , 0.5135723 , 0.51724875, 0.5121017 , 0.51994485, 0.4375919 , 0.44715074, 0.48734683, 0.38465074,       0.41945466, 0.44690564, 0.46038604, 0.47852328, 0.49543506, 0.6795037 , 0.69224876, 0.6493566 , 0.5130821 , 0.5020527 ,       0.5135723 , 0.517739  , 0.3532782 , 0.38146445, 0.4035233 , 0.4287684 , 0.44837624, 0.45621938, 0.46920955, 0.4750919 ,       0.48170957, 0.48563114, 0.5035233 , 0.677788  , 0.7025429 , 0.6971507 , 0.5133272 , 0.5155331 , 0.3897978 , 0.4145527 ,       0.43587622, 0.44862133, 0.45352328, 0.46847427, 0.47386643, 0.48440564, 0.47999388, 0.49200368, 0.49053308, 0.5018076 ,       0.50719976, 0.64739585, 0.69445467, 0.6905331 , 0.42068014, 0.43857232, 0.4542586 , 0.45793504, 0.47411153, 0.48072916,       0.47852328, 0.4890625 , 0.49420956, 0.49273896, 0.5018076 , 0.49641544, 0.5135723 , 0.5182292 , 0.5121017 , 0.5518076 ,       0.44764093, 0.4481311 , 0.45499387, 0.46087623, 0.46063113, 0.4733762 , 0.49518996, 0.50695467, 0.4885723 , 0.49396446,       0.49715075, 0.50695467, 0.5133272 , 0.50499386, 0.5179841 , 0.515288  , 0.45474878, 0.45744485, 0.46014094, 0.46651348,       0.462837  , 0.48440564, 0.5059743 , 0.4986213 , 0.48587623, 0.49788603, 0.5010723 , 0.50842524, 0.50474876, 0.5211703 ,       0.51063114, 0.5150429 , 0.44739583, 0.45695466, 0.46577817, 0.46871936, 0.47460172, 0.48170957, 0.49249387, 0.49935663,       0.49322918, 0.52264094, 0.5020527 , 0.49960172, 0.5018076 , 0.5059743 , 0.512837  , 0.51896447};
   
    for (int inputVar=0; inputVar<inputSize; inputVar++){
        l[inputVar] = img[inputVar]-delta;
        u[inputVar] = img[inputVar]+delta;
    }

    struct Matrix input_upper = {u,1,nnet->inputSize};
    struct Matrix input_lower = {l,1,nnet->inputSize};

    struct Interval input_interval = {input_lower, input_upper};

    float grad_upper[inputSize], grad_lower[inputSize];
    struct Interval grad_interval = {
                (struct Matrix){grad_upper, 1, inputSize},
                (struct Matrix){grad_lower, 1, inputSize}
            };

    normalize_input_interval(nnet, &input_interval);

    float o[nnet->outputSize];
    struct Matrix output = {o, outputSize, 1};

    float o_upper[nnet->outputSize], o_lower[nnet->outputSize];
    struct Interval output_interval = {
                (struct Matrix){o_lower, outputSize, 1},
                (struct Matrix){o_upper, outputSize, 1}
            };
    
    int n = 0;
    int feature_range_length = 0;
    int split_feature = -1;
    int depth = 0;
    
    printf("running property %d with network %s\n",\
                PROPERTY, FULL_NET_PATH);
    printf("input ranges:\n");

    printMatrix(&input_upper);
    printMatrix(&input_lower);


    for (int i=0;i<inputSize;i++) {

        if (input_interval.upper_matrix.data[i] <\
                input_interval.lower_matrix.data[i]) {
            printf("wrong input!\n");
            exit(0);
        }

        if(input_interval.upper_matrix.data[i] !=\
                input_interval.lower_matrix.data[i]){
            n++;
        }

    }

    feature_range_length = n;
    int *feature_range = (int*)malloc(n*sizeof(int));

    for (int i=0, n=0;i<nnet->inputSize;i++) {
        if(input_interval.upper_matrix.data[i] !=\
                input_interval.lower_matrix.data[i]){
            feature_range[n] = i;
            n++;
        }
    }
    
    gettimeofday(&start, NULL);
    int isOverlap = 0;
    float avg[100] = {0};

    if (CHECK_ADV_MODE) {
        printf("check mode: CHECK_ADV_MODE\n");
        isOverlap = direct_run_check(nnet,\
                &input_interval, &output_interval,\
                &grad_interval, depth, feature_range,\
                feature_range_length, split_feature);
    
    }
    else {
        printf("check mode: NORMAL_CHECK_MODE\n");

        for (int i=0;i<1;i++) {
            //forward_prop_interval_equation(nnet,\
                    &input_interval, &output_interval,\
                    &grad_interval);
            isOverlap = direct_run_check(nnet,\
                    &input_interval, &output_interval,\
                    &grad_interval, depth, feature_range,\
                    feature_range_length, split_feature);
        }

    }
    
    gettimeofday(&finish, NULL);
    time_spent = ((float)(finish.tv_sec - start.tv_sec) *\
            1000000 + (float)(finish.tv_usec - start.tv_usec)) /\
            1000000;

    if (isOverlap == 0 && adv_found == 0) {
        printf("\nNo adv!\n");
    }

    
    printf("time: %f \n\n\n", time_spent);

    destroy_network(nnet);
    free(feature_range);

}