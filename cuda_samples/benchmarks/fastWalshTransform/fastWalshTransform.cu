/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * Walsh transforms belong to a class of generalized Fourier transformations.
 * They have applications in various fields of electrical engineering
 * and numeric theory. In this sample we demonstrate efficient implementation
 * of naturally-ordered Walsh transform
 * (also known as Walsh-Hadamard or Hadamard transform) in CUDA and its
 * particular application to dyadic convolution computation.
 * Refer to excellent Jorg Arndt's "Algorithms for Programmers" textbook
 * http://www.jjj.de/fxt/fxtbook.pdf (Chapter 22)
 *
 * Victor Podlozhnyuk (vpodlozhnyuk@nvidia.com)
 */



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_functions.h>
#include <helper_cuda.h>


////////////////////////////////////////////////////////////////////////////////
// Reference CPU FWT
////////////////////////////////////////////////////////////////////////////////
extern"C" void fwtCPU(float *h_Output, float *h_Input, int log2N);
extern"C" void slowWTcpu(float *h_Output, float *h_Input, int log2N);
extern "C" void dyadicConvolutionCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int log2dataN,
    int log2kernelN
);


////////////////////////////////////////////////////////////////////////////////
// GPU FWT
////////////////////////////////////////////////////////////////////////////////
#include "fastWalshTransform_kernel.cuh"



////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int log2Kernel = 7;
const   int log2Data = 23;

const int   dataN = 1 << log2Data;
const int kernelN = 1 << log2Kernel;

const int   DATA_SIZE = dataN   * sizeof(float);
const int KERNEL_SIZE = kernelN * sizeof(float);

const double NOPS = 3.0 * (double)dataN * (double)log2Data / 2.0;



////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    float *h_Data,
          *h_Kernel,
          *h_ResultCPU,
          *h_ResultGPU;

    float *d_Data,
          *d_Kernel;

    double gpuTime;

    StopWatchInterface *hTimer = NULL;
    int i;
    int rep;

    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    findCudaDevice(argc, (const char **)argv);

    sdkCreateTimer(&hTimer);

    printf("Initializing data...\n");
    printf("...allocating CPU memory\n");
    h_Kernel    = (float *)malloc(KERNEL_SIZE);
    h_Data      = (float *)malloc(DATA_SIZE);
    h_ResultCPU = (float *)malloc(DATA_SIZE);
    h_ResultGPU = (float *)malloc(DATA_SIZE);
    printf("...allocating GPU memory\n");
    checkCudaErrors(cudaMalloc((void **)&d_Kernel, DATA_SIZE));
    checkCudaErrors(cudaMalloc((void **)&d_Data,   DATA_SIZE));

    printf("...generating data\n");
    printf("Data length: %i; kernel length: %i\n", dataN, kernelN);
    srand(2007);

    for (i = 0; i < kernelN; i++)
    {
        h_Kernel[i] = (float)rand() / (float)RAND_MAX;
    }

    for (i = 0; i < dataN; i++)
    {
        h_Data[i] = (float)rand() / (float)RAND_MAX;
    }

    checkCudaErrors(cudaMemset(d_Kernel, 0, DATA_SIZE));
    checkCudaErrors(cudaMemcpy(d_Kernel, h_Kernel, KERNEL_SIZE, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Data,   h_Data,     DATA_SIZE, cudaMemcpyHostToDevice));

    printf("Running GPU dyadic convolution using Fast Walsh Transform...\n");
    checkCudaErrors(cudaDeviceSynchronize());

    time_t t;
    struct tm * lt;
    time(&t);
    lt = localtime(&t);
    printf("%d-%d-%d %d:%d:%d\n" ,lt->tm_year+1900, lt->tm_mon+1, lt->tm_mday, lt->tm_hour, lt->tm_min, lt->tm_sec);

    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    for (rep = 0; rep < 54000; rep++)
    {
        fwtBatchGPU(d_Data, 1, log2Data);
        fwtBatchGPU(d_Kernel, 1, log2Data);
        modulateGPU(d_Data, d_Kernel, dataN);
        fwtBatchGPU(d_Data, 1, log2Data);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    time(&t);
    lt = localtime(&t);
    printf("%d-%d-%d %d:%d:%d\n" ,lt->tm_year+1900, lt->tm_mon+1, lt->tm_mday, lt->tm_hour, lt->tm_min, lt->tm_sec);
    printf("GPU time: %f ms; GOP/s: %f\n", gpuTime, 30000 * NOPS / (gpuTime * 0.001 * 1E+9));


    printf("Shutting down...\n");
    sdkDeleteTimer(&hTimer);
    checkCudaErrors(cudaFree(d_Data));
    checkCudaErrors(cudaFree(d_Kernel));
    free(h_ResultGPU);
    free(h_ResultCPU);
    free(h_Data);
    free(h_Kernel);

}
