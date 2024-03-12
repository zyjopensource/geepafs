/* MIT License

Copyright (c) 2023 Yijia Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

 * GPU Energy-Efficient and Performance-Assured Frequency Scaling (GEEPAFS) policy program.
 * This program reads NVIDIA GPU metrics and tunes the GPU frequency using the NVML tool.
 
 * Select the correct GPU type by editing the following "#define" lines.
 * Run "make" to compile this code. Note that "CUDA_PATH" in Makefile may need to be updated.
 * Run with default settings by the command "sudo ./dvfs mod Assure p90".
 * Root privileges are necessary in applying frequency tuning.
 * This program runs endlessly. Press ctrl-c to stop. Frequency is reset automatically at stop.
 * "verbose" variable can be initialized as true to output more details.
 */


// Please select one of the following MACHINE choices by editing the "#define" lines.:
// ("v100-maxq" is V100 GPU with TDP 163 W; "v100-300w" is V100 GPU with TDP 300 W; "a100-insp" is A100 GPU with TDP 400 W.)

//#define MACHINE "v100-maxq"
#define MACHINE "v100-300w"
//#define MACHINE "a100-insp"

// Other GPU types are not directly supported so far. To use this code for another GPU type,
// constants minSetFreq, freqAvgEff, maxFreq, setMemFreq, numAvailableFreqs, numProbFreq, probFreqs,
// and the function getAvailableFreqs() need to be updated.

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <nvml.h>
#include <sys/time.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <math.h>
#include <time.h>

static volatile int keepRunning = 1;

void intHandler(int dummy) // function used for the interruptable loop.
{
    keepRunning = 0;
}

double max(double a, double b)
{
    if (a >= b)
        return a;
    else
        return b;
}

double min(double a, double b)
{
    if (a >= b)
        return b;
    else
        return a;
}

void linearRegression(int num, double* X, double* Y, double* slope, double* intercept, double* regErr) // linear regression.
{
    int k;
    double sumx=0, sumxsq=0, sumy=0, sumxy=0, sumysq=0, div, xd, yd, a, b;
    for (k=0; k<num; k++)
    {
        xd = X[k];
        yd = Y[k];
        sumx += xd;
        sumxsq += xd*xd;
        sumy += yd;
        sumxy += xd*yd;
        sumysq += yd*yd;
    }
    div = num*sumxsq - sumx*sumx;
    a = (num*sumxy-sumx*sumy) / div;
    b = (sumy*sumxsq-sumx*sumxy) / div;
    *slope = a;
    *intercept = b;
    *regErr = sumysq + a*a*sumxsq + num*b*b - 2*a*sumxy - 2*b*sumy + 2*a*b*sumx;
}

void foldlineRegression(int xc, int num1, double* X1, double* Y1, int num2, double* X2, double* Y2, double* slope1, double* intercept1, double* slope2, double* intercept2, double* regErr) // Fold-line regression with the assumption that the fold-point's x position is at xc.
{
    int i, j;
    double err=0, a1, b1, a2, b2, H, c11, c12, c13, c14, c21, c22, c23, c24, c31, c32, c33, c34, n;
    double sum1_x=0, sum1_y=0, sum2_x=0, sum2_y=0, sum2_xsq=0, sum2_xy=0, sum1_xsq=0, sum1_xy=0;
    n = num1 + num2;
    for (i=0; i<num1; i++)
    {
        sum1_x += X1[i];
        sum1_y += Y1[i];
        sum1_xsq += X1[i]*X1[i];
        sum1_xy += X1[i]*Y1[i];
    }
    for (j=0; j<num2; j++)
    {
        sum2_x += X2[j];
        sum2_y += Y2[j];
        sum2_xsq += X2[j]*X2[j];
        sum2_xy += X2[j]*Y2[j];
    }

    c11 = sum1_xsq + num2*xc*xc;
    c12 = xc*sum2_x - num2*xc*xc;
    c13 = sum1_x + xc*num2;
    c14 = - sum1_xy - sum2_y*xc;
    c21 = xc*sum2_x - num2*xc*xc;
    c22 = sum2_xsq - 2*xc*sum2_x + num2*xc*xc;
    c23 = sum2_x - num2*xc;
    c24 = - sum2_xy + xc*sum2_y;
    c31 = sum1_x + num2*xc;
    c32 = sum2_x - num2*xc;
    c33 = n;
    c34 = - sum1_y - sum2_y;

    H = c11*c22*c33 + c12*c23*c31 + c21*c32*c13 - c13*c22*c31 - c12*c21*c33 - c11*c23*c32;
    if (H == 0)
    {
        a1 = -1; b1 = -2; a2 = -3; b2 = -4; err = 12345678;
    }
    else
    {
        a1 = -(c14*c22*c33 + c12*c23*c34 + c13*c24*c32 - c13*c22*c34 - c12*c24*c33 - c23*c32*c14) / H;
        a2 = -(c11*c24*c33 + c21*c34*c13 + c14*c23*c31 - c13*c31*c24 - c11*c23*c34 - c33*c14*c21) / H;
        b1 = -(c11*c22*c34 + c21*c32*c14 + c12*c24*c31 - c22*c14*c31 - c12*c21*c34 - c11*c32*c24) / H;
        b2 = xc*(a1-a2) + b1;
        for (i=0; i<num1; i++)
        {
            err += (a1*X1[i] + b1 - Y1[i])*(a1*X1[i] + b1 - Y1[i]);
        }
        for (j=0; j<num2; j++)
        {
            err += (a2*X2[j] + b2 - Y2[j])*(a2*X2[j] + b2 - Y2[j]);
        }
    }
    *slope1 = a1;
    *intercept1 = b1;
    *slope2 = a2;
    *intercept2 = b2;
    *regErr = err;
}

void getAvailableFreqs(int* availableFreqs, int numAvailableFreqs) // get all available frequency values.
{
    // Run "nvidia-smi -q -d SUPPORTED_CLOCKS" to get available frequencies and update this function if needed.
    if (strcmp(MACHINE, "v100-maxq") == 0)
    {
        int idx = 1;
        int freq = 135;
        bool seven = true;
        availableFreqs[0] = freq;
        while (freq <= 1440)
        {
            if (seven)
            {
                freq += 7;
                seven = false;
            }
            else
            {
                freq += 8;
                seven = true;
            }
            if (freq <= 1440)
            {
                availableFreqs[idx] = freq;
                idx += 1;
            }
        }
    }
    else if (strcmp(MACHINE, "v100-300w") == 0)
    {
        int idx = 1;
        int freq = 135;
        bool seven = true;
        availableFreqs[0] = freq;
        while (freq <= 1530)
        {
            if (seven)
            {
                freq += 7;
                seven = false;
            }
            else
            {
                freq += 8;
                seven = true;
            }
            if (freq <= 1530)
            {
                availableFreqs[idx] = freq;
                idx += 1;
            }
        }
    }
    else if (strcmp(MACHINE, "a100-insp") == 0)
    {
	    int idx = 1;
	    int freq = 210;
	    availableFreqs[0] = freq;
	    while (freq <= 1410)
	    {
	        freq += 15;
	        if (freq <= 1410)
	        {
		    availableFreqs[idx] = freq;
		    idx += 1;
	        }
	    }
    }
}

int main(int argc, char* argv[])
{
    // Adjustable arguments.
    unsigned int minSetFreq;// The lower bound for setting frequency.
    unsigned int freqAvgEff;// the globally most power efficient frequency based on experiments from many apps.
    unsigned int maxFreq;// max freq supported.
    unsigned int setMemFreq;// the only available memory freq value on this machine.
    int numAvailableFreqs;// number of available freqs supported by this machine.
    int numProbFreq; // number of freqs to be probed in the probing phase. the freq values are in variable probFreqs.
    int* const probFreqs = (int*)malloc(sizeof(int)*20);// reserve enough space for probing freqs.

    // Run "nvidia-smi -q -d SUPPORTED_CLOCKS" to get available frequencies and update the following parameters if needed.
    if (strcmp(MACHINE, "v100-maxq") == 0)
    {
        minSetFreq = 855; freqAvgEff = 855; maxFreq = 1440; setMemFreq = 810; numAvailableFreqs = 175;
        numProbFreq = 4;
        probFreqs[0] = 855; // frequency values for probing.
        probFreqs[1] = 1050;
        probFreqs[2] = 1245;
        probFreqs[3] = 1440;
    }
    else if (strcmp(MACHINE, "v100-300w") == 0)
    {
        minSetFreq = 952; freqAvgEff = 952; maxFreq = 1530; setMemFreq = 877; numAvailableFreqs = 187;
        numProbFreq = 4;
        probFreqs[0] = 952; // frequency values for probing.
        probFreqs[1] = 1147;
        probFreqs[2] = 1335;
        probFreqs[3] = 1530;
    }
    else if (strcmp(MACHINE, "a100-insp") == 0)
    {
        minSetFreq = 1110; freqAvgEff = 1110; maxFreq = 1410; setMemFreq = 1593; numAvailableFreqs = 81;
        numProbFreq = 4;
        probFreqs[0] = 1110; // frequency values for probing.
        probFreqs[1] = 1215;
        probFreqs[2] = 1320;
        probFreqs[3] = 1410;
    }

    const char *allArg = "mod for modulate";
    const char *argAbbre= "mod";
    printf("Apply policy: %s\n",argv[2]);
    //char *freqsetAlg = "Assure";// NVboost, MaxFreq, EfficientFix, UtilizScale.
    char *freqsetAlg = argv[2];
    double perfThres = 0.90;// key parameter in Assure. performance should not drop below this percentage when doing DVFS.
    if (strcmp(argv[1], "mod") == 0 && strcmp(freqsetAlg, "Assure") == 0)
    {
        if (strcmp(argv[3], "p95") == 0)
            perfThres = 0.95;
        if (strcmp(argv[3], "p90") == 0)
            perfThres = 0.90;
        if (strcmp(argv[3], "p85") == 0)
            perfThres = 0.85;
    }
    const bool useFreqCap = true;// whether set an upper bound.
    const bool useRegression = true;
    const int loopDelay = 200;// minimal interval of each loop in milliseconds. Used in multiple policies.
    const double probDelay = 15;// interval between two probing phase in seconds.
    const int numProbRep = 2; // reptition of each frequency point in the probing phase.
    const double regErrThres = 100; // average regression error threshold per point, beyond which regression model is discarded.

    const int probInterval = 20;// used in baseline policies.
    const int movingAvg_windowSize = 16;// window size for calcuting the moving avg/std.

    // Utility variables.
    const bool onlySetFreqForOne = false;// default false. If true, only set freq for one gpu to avoid affecting other jobs.
    const int onlySetGPUIdx = 1;// effective only when onlySetFreqForOne is true.
    const bool printUtil = true;// default true.
    const bool onlySetAppFreq = true;// default true. true - nvmlDeviceSetApplicationsClocks(); false - nvmlDeviceSetGpuLockedClocks().
    const bool verbose = false;// default false.
    const bool skipSetFreq = false;// default false. true is only used to measure the cost of this tool.

    // Dependent variables. No need to change.
    nvmlReturn_t result;
    nvmlDevice_t device;
    nvmlUtilization_t util;
    nvmlClockType_t freq;
    unsigned int power, device_count, i, k, setFreq, turn, turn_Opt, ifreq;// warning: unsigned int should not loop from high to low.
    struct timeval starttime, endtime;
    long unsigned int duration, addTime;
    long unsigned int accumuTime = 0;
    int j, iAvail, iprob, reminder, mostEfficiFreq, max_gmem_freq;
    int recordNum=0, cycle=0, lastprobPhase=0, idx1=0, idx2=0, idx_oldest=0, changeDetect_delay=0;
    int probPhase = numProbFreq * numProbRep; // probing start at the beginning. This variable is not constant.
    const int numProbRec = numProbFreq * numProbRep;
    double slope_Opt, slope1, slope2, slope1_Opt, slope2_Opt, intercept_Opt, intercept1, intercept2, intercept1_Opt, intercept2_Opt, sumy, regErr, regErr1, regErr2, regErrMin, freq_perfBound, freq_cross, variance, sum_gutil, mostEffici, criticalPerf, thisCap, max_gmem, freqBound, freqPerf, freqOpt, freqEff, f, c0, c1, c2, c3;
    bool optimalFound, process_exist, freqsetHappen, skipmodel, applyFreqSet;
    bool initialLoop=true;
    time_t t;
    struct tm * lt;

    // Initialize.
    printf("MACHINE %s\n", MACHINE);
    printf("GPU freqset tool start..\n");
    if (argc < 2)
    {
        printf("Error: Needs argument: %s\n", allArg);
        return 1;
    }
    if (strstr(argAbbre, argv[1]) == NULL)
    {
        printf("Error: Only the following arguments are allowed: %s\n", argAbbre);
        return 1;
    }
    result = nvmlInit_v2();
    if (NVML_SUCCESS != result)
    { 
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return 1;
    }
    result = nvmlDeviceGetCount(&device_count);
    if (NVML_SUCCESS != result)
    { 
        printf("Failed to query GPU count: %s\n", nvmlErrorString(result));
        goto Error;
    }

    // Initialize arrays.
    int* const optimizedFreqs = (int*)malloc(sizeof(int)*device_count);
    int** const gpuUtils = (int**)malloc(sizeof(int*)*device_count);// a 2-d array. Each row is for a certain gpu.
    int** const gpuUtils_sq = (int**)malloc(sizeof(int*)*device_count);// a 2-d array. Each row is for a certain gpu.
    double** const gmemUtils = (double**)malloc(sizeof(double*)*device_count);// a 2-d array. Each row is for a certain gpu.
    double** const gPowers = (double**)malloc(sizeof(double*)*device_count);// a 2-d array. Each row is for a certain gpu.
    double* const avg_gmemUtils = (double*)malloc(sizeof(double)*numProbFreq);// record the average gmemUtil for each probing frequency.
    double* const avg_gPowers = (double*)malloc(sizeof(double)*numProbFreq);// record the average gPower for each probing frequency.
    double* const modelPerf = (double*)malloc(sizeof(double)*numProbFreq);// record model-estimated performance.
    double* const powerEffici = (double*)malloc(sizeof(double)*numProbFreq);// record power efficiency.
    for (i = 0; i < device_count; i++)
    {
        optimizedFreqs[i] = maxFreq;// initialized value.
        gpuUtils[i] = (int*)malloc(sizeof(int)*movingAvg_windowSize);// to record gpu util for change detection.
        for (j = 0; j < movingAvg_windowSize; j++)
        {
            gpuUtils[i][j] = 0;
        }
        gpuUtils_sq[i] = (int*)malloc(sizeof(int) * movingAvg_windowSize);// to record square of gpu util.
        for (j = 0; j < movingAvg_windowSize; j++)
        {
            gpuUtils_sq[i][j] = 0;
        }
        gmemUtils[i] = (double*)malloc(sizeof(double) * numProbRec);
        for (j = 0; j < numProbRec; j++)
        {
            gmemUtils[i][j] = 0;
        }
        gPowers[i] = (double*)malloc(sizeof(double) * numProbRec);
        for (j = 0; j < numProbRec; j++)
        {
            gPowers[i][j] = 0;
        }
    }
    double* const x = (double*)malloc(sizeof(double)*numProbRec);
    double* const y = (double*)malloc(sizeof(double)*numProbRec);
    double* const x1 = (double*)malloc(sizeof(double)*numProbRec);
    double* const y1 = (double*)malloc(sizeof(double)*numProbRec);
    double* const x2 = (double*)malloc(sizeof(double)*numProbRec);
    double* const y2 = (double*)malloc(sizeof(double)*numProbRec);
    double* const x3 = (double*)malloc(sizeof(double)*numProbRec);
    double* const y3 = (double*)malloc(sizeof(double)*numProbRec);
    for (i = 0; i < numProbRec; i++)
    {
        x[i] = 0; y[i] = 0; x1[i] = 0; y1[i] = 0; x2[i] = 0; y2[i] = 0; x3[i] = 0; y3[i] = 0;
    }
    double* const gutil_moving_avg = (double*)malloc(sizeof(double)*device_count);
    double* const gutil_moving_sqsum = (double*)malloc(sizeof(double)*device_count);
    double* const gutil_moving_std = (double*)malloc(sizeof(double)*device_count);
    double* const freqCap = (double*)malloc(sizeof(double)*device_count);
    for (i = 0; i < device_count; i++)
    {
        gutil_moving_avg[i] = 0;
        gutil_moving_sqsum[i] = 0;
    }
    int* const availableFreqs = (int*)malloc(sizeof(int)*numAvailableFreqs);
    getAvailableFreqs(availableFreqs, numAvailableFreqs);
    if (verbose)
    {
        printf("Available frequencies:\n");
        for (i = 0; i < numAvailableFreqs; i++)
            printf("%d\t", availableFreqs[i]);
        printf("\n");
    }
    double* const allModelPerf = (double*)malloc(sizeof(double)*numAvailableFreqs);
    double* const allModelPower = (double*)malloc(sizeof(double)*numAvailableFreqs);
    double* const allPowerEffici = (double*)malloc(sizeof(double)*numAvailableFreqs);
    for (i = 0; i < numAvailableFreqs; i++)
    {
        allModelPerf[i] = -1;
        allModelPower[i] = -1;
        allPowerEffici[i] = -1;
    }
    if (verbose)
        printf("Warning: verbose set as true.\n");
    if (skipSetFreq)
        printf("Warning: skipSetFreq set as true.\n");
    if (onlySetFreqForOne)
        printf("Warning: onlySetFreqForOne set as true.\n");

    // Reset GPU clocks.
    printf("Reset GPU frequency for: ");
    for (i = 0; i < device_count; i++)
    {
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (NVML_SUCCESS != result)
        { 
            printf("Failed to get handle for GPU %u: %s\n", i, nvmlErrorString(result));
            goto Error;
        }

        result = nvmlDeviceResetGpuLockedClocks(device);
        if (NVML_ERROR_NO_PERMISSION == result)
        {
            printf("\t\t Error: Need root privileges: %s\n", nvmlErrorString(result));
            goto Error;
        }
        else if (NVML_ERROR_NOT_SUPPORTED == result)
            printf("\t\t Operation not supported.\n");
        else if (NVML_SUCCESS != result)
        {
            printf("\t\t Failed to reset locked frequency for GPU %u: %s\n", i, nvmlErrorString(result));
            goto Error;
        } 

        result = nvmlDeviceResetApplicationsClocks(device);
        if (NVML_ERROR_NO_PERMISSION == result)
            printf("\t\t Need root privileges: %s\n", nvmlErrorString(result));
        else if (NVML_ERROR_NOT_SUPPORTED == result)
            printf("\t\t Operation not supported.\n");
        else if (NVML_SUCCESS != result)
        {
            printf("\t\t Failed to reset application frequency for GPU %u: %s\n", i, nvmlErrorString(result));
            goto Error;
        } 
        else if (NVML_SUCCESS == result)
        {
            printf("device %u. ", i);
        }
    }
    printf("\n");

    // main loop.
    signal(SIGINT, intHandler);
    printf("Main loop start..\n");
    while (keepRunning) // press ctrl+c can break this loop.
    {
        gettimeofday(&starttime, NULL);
        if (printUtil)
        {
            time(&t);
            lt = localtime(&t);
            printf("%d-%d-%d %d:%d:%d, " ,lt->tm_year+1900, lt->tm_mon+1, lt->tm_mday, lt->tm_hour, lt->tm_min, lt->tm_sec);
        }

        // loop through each GPU.
        for (i = 0; i < device_count; i++)
        {
            result = nvmlDeviceGetHandleByIndex(i, &device);
            if (NVML_SUCCESS != result)
            { 
                printf("Failed to get handle for GPU %u: %s\n", i, nvmlErrorString(result));
                goto Error;
            }

            // get gpu utilization rate (including gmem bandwidth util).
            result = nvmlDeviceGetUtilizationRates(device, &util);
            if (NVML_SUCCESS != result)
            { 
                printf("Failed to get utilization rate for GPU %u: %s\n", i, nvmlErrorString(result));
                goto Error;
            }

            // get gpu frequency.
            result = nvmlDeviceGetClockInfo(device, 1, &freq);// 1 refers to SM domain.
            if (NVML_SUCCESS != result)
            { 
                printf("Failed to get clock frequency for GPU %u: %s\n", i, nvmlErrorString(result));
                goto Error;
            }

            // get gpu power usage.
            result = nvmlDeviceGetPowerUsage(device, &power);
            if (NVML_SUCCESS != result)
            {
                printf("Failed to get power usage for GPU %u: %s\n", i, nvmlErrorString(result));
                goto Error;
            }

            // set GPU frequency.
            // MaxFreq policy.
            if (strcmp(freqsetAlg, "MaxFreq") == 0)
            {
                setFreq = maxFreq;
                applyFreqSet = initialLoop;
            }
            // EfficientFix policy.
            if (strcmp(freqsetAlg, "EfficientFix") == 0)
            {
                setFreq = freqAvgEff;
                applyFreqSet = initialLoop;
            }
            // NVboost policy. Using the default policy, not applying user freq set.
            if (strcmp(freqsetAlg, "NVboost") == 0)
            {
                setFreq = freqAvgEff;
                applyFreqSet = false;
            }
            // UtilizScale policy.
            if (strcmp(freqsetAlg, "UtilizScale") == 0)
            {
                if (cycle == 1)
                {
                    // Prob the utilization at max frequency.
                    setFreq = maxFreq;
                    applyFreqSet = true;
                }
                else if (cycle == 2)
                {
                    // Set freq proportional to gpu util, bounded by minSetFreq from below.
                    optimizedFreqs[i] = (int)max((double)minSetFreq, (double)util.gpu/100*(double)maxFreq);
                    // Loop through all available frequency values to find the nearest larger one.
                    for (iAvail = numAvailableFreqs-1; iAvail >= 0; iAvail--)
                    {
                        if (availableFreqs[iAvail] < optimizedFreqs[i])
                        {
                            if (iAvail < numAvailableFreqs-1)
                                optimizedFreqs[i] = availableFreqs[iAvail+1];
                            else
                                optimizedFreqs[i] = availableFreqs[iAvail];
                            break;
                        }
                    }
                    setFreq = optimizedFreqs[i];
                    applyFreqSet = true;
                }
                else
                {
                    setFreq = optimizedFreqs[i];
                    applyFreqSet = false;
                }
            }
            // Assure policy.
            if (strcmp(freqsetAlg, "Assure") == 0)
            {
                // Calculate moving average. And record gpu utilization into 2-d array **gpuUtils.
                // Calculating average should start from the oldest value. idx_oldest markes the oldest position.
                gutil_moving_avg[i] = gutil_moving_avg[i] - (double)gpuUtils[i][idx_oldest] / movingAvg_windowSize + (double)util.gpu / movingAvg_windowSize;// update the moving average.
                gutil_moving_sqsum[i] = gutil_moving_sqsum[i] - (double)gpuUtils_sq[i][idx_oldest] + (double)util.gpu * (double)util.gpu;// update the moving sum of square of gpuUtil.
                variance = gutil_moving_sqsum[i]/movingAvg_windowSize - gutil_moving_avg[i]*gutil_moving_avg[i];
                if (variance > 0)
                    gutil_moving_std[i] = sqrt(variance);
                else
                    gutil_moving_std[i] = 0;
                gpuUtils[i][idx_oldest] = util.gpu;
                gpuUtils_sq[i][idx_oldest] = util.gpu * util.gpu;
                if (i == device_count-1)
                {
                    // forward idx_oldest by 1 position.
                    if (idx_oldest < movingAvg_windowSize-1)
                        idx_oldest += 1;
                    else
                        idx_oldest = 0;
                }

                if (lastprobPhase > 0)// lastprobPhase starts from numProbRec.
                {
                    // During probing phase, record gpu memory bandwidth utilization into 2-d array **gmemUtils.
                    // Record gpu power usage into **gPowers.
                    // Index of gmemUtils[i] should start from 0.
                    // Be careful that the recorded util values corresponds to the last frequency setting.
                    if (verbose && i==0)
                        printf("lastprobPhase %d, ", lastprobPhase);
                    gmemUtils[i][numProbRec-lastprobPhase] = (double)util.memory;
                    gPowers[i][numProbRec-lastprobPhase] = (double)power/1000;// on V100, power is in mW.

                    if (useFreqCap)
                    {
                        // calculate the freq cap according to the current gpu util and gpu freq.
                        thisCap = (double)freq / ((1-perfThres)*((double)freq/(double)maxFreq+100/max(1,(double)util.gpu)-1) + (double)freq/(double)maxFreq);// max(1,) is used to avoid division by 0.
                        //thisCap = perfThres*(double)freq*(double)util.gpu/100;
                        // freqCap[i] records the largest cap during probing.
                        if (lastprobPhase == numProbRec)
                        {
                            freqCap[i] = thisCap;
                        }
                        else
                        {
                            if (thisCap > freqCap[i])
                                freqCap[i] = thisCap;
                        }
                    }
                }

                if (probPhase > 0)
                {
                    // in probing phase, force changing gpu freqs to prob the response of gpu utils.
                    iprob = numProbRec - probPhase; // iprob start at 0 and increase.
                    reminder = iprob % (2*numProbFreq);
                    if (reminder < numProbFreq)
                        setFreq = probFreqs[reminder];
                    else
                        setFreq = probFreqs[2*numProbFreq-1-reminder];
                }
                else if (probPhase == 0)
                {
                    // keep the last freq setting.
                    iprob = numProbRec - 1;
                    reminder = iprob % (2*numProbFreq);
                    if (reminder < numProbFreq)
                        setFreq = probFreqs[reminder];
                    else
                        setFreq = probFreqs[2*numProbFreq-1-reminder];
                }
                else
                {
                    if (skipSetFreq)
                        setFreq = maxFreq;// only for measuring policy cost.
                    else
                        setFreq = optimizedFreqs[i];// calculated when probPhase==0.
                }
                if (probPhase >= -1)
                    applyFreqSet = true;
                else
                    applyFreqSet = false;// not apply freq set to reduce delay.
            }// end if Assure.

            // Execute frequency set.
            // Note: Avoiding unnecessary freqset can significantly reduce delay, from 90 ms to 13 ms.
            // Note: When power is high, actual freq may be consistently lower than setFreq due to thermal throttling.
            if (applyFreqSet)
            {
                freqsetHappen = false;
                if (onlySetAppFreq)
                {
                    if (onlySetFreqForOne)
                    {
                        if (onlySetGPUIdx == i)
                            result = nvmlDeviceSetApplicationsClocks(device, setMemFreq, setFreq);
                        else
                            result = NVML_SUCCESS;
                    }
                    else
                        result = nvmlDeviceSetApplicationsClocks(device, setMemFreq, setFreq);
                    freqsetHappen = true;
                }
                else
                {
                    result = nvmlDeviceSetGpuLockedClocks(device, setFreq, setFreq);
                    freqsetHappen = true;
                }

                if (freqsetHappen)
                {
                    if (NVML_ERROR_NO_PERMISSION == result)
                        printf("\t\t Error: Need root privileges: %s\n", nvmlErrorString(result));
                    else if (NVML_ERROR_NOT_SUPPORTED == result)
                        printf("\t\t Operation not supported.\n");
                    else if (NVML_SUCCESS != result)
                    {
                        printf("\t\t Failed to set frequency for GPU %u: %s\n", i, nvmlErrorString(result));
                        goto Error;
                    } 
                    else if (NVML_SUCCESS == result && printUtil)
                    {
                        printf("%u, %u, %u, %u, %u, ", util.gpu, util.memory, power, freq, setFreq);
                    }
                }
            }
            else
            {
                if (printUtil)
                {
                    printf("%u, %u, %u, %u, -1, ", util.gpu, util.memory, power, freq);// -1 is a flag for this case.
                }
            }
        }// loop all GPU ends.

        // In Assure, if just finished probing phase, fit the performance model and calculate the optimized freq.
        if (strcmp(argv[1], "mod") == 0 && strcmp(freqsetAlg, "Assure") == 0)
        {
            if (probPhase == 0)// i.e., when just finished probing.
            {
                // calculate the freq cap according to gpu util.
                if (verbose && useFreqCap)
                {
                    printf("\nFrequency cap according to utilization:");
                    for (i = 0; i < device_count; i++)
                    {
                        printf("\t%.f", freqCap[i]);
                    }
                    printf("\n");
                }

                // print the gemUtils array.
                if (verbose)
                {
                    for (i = 0; i < device_count; i++)
                    {
                        printf("Device %u mem bw util: ", i);
                        for (j = 0; j < numProbRec; j++)
                        {
                            if (j > 0 && j % numProbFreq == 0)
                                printf("| ");
                            printf("%.lf ", gmemUtils[i][j]);
                        }
                        printf("\n");
                    }
                }

                for (i = 0; i < device_count; i++)
                {
                    // Calculate avg_gmemUtils and avg_gPowers.
                    // construct the x, y input for the single linear model.
                    // Numbers should be converted into double.
                    sumy = 0;
                    for (j = 0; j < numProbFreq; j++)
                    {
                        avg_gmemUtils[j] = 0;
                        avg_gPowers[j] = 0;
                    }
                    for (j = 0; j < numProbRec; j++)
                    {
                        reminder = j % (2*numProbFreq);
                        y[j] = gmemUtils[i][j];
                        sumy += y[j];
                        if (reminder < numProbFreq)
                        {
                            x[j] = (double)probFreqs[reminder];
                            avg_gmemUtils[reminder] += y[j] / (double)numProbRep;
                            avg_gPowers[reminder] += gPowers[i][j] / (double)numProbRep;
                        }
                        else
                        {
                            x[j] = (double)probFreqs[2*numProbFreq-1-reminder];
                            avg_gmemUtils[2*numProbFreq-1-reminder] += y[j] / (double)numProbRep;
                            avg_gPowers[2*numProbFreq-1-reminder] += gPowers[i][j] / (double)numProbRep;
                        }
                    }

                    // Fit the model with fold-line regression.
                    if (useRegression)
                    {
                        // optimize frequency when all the gmem util is nonzero.
                        if (sumy > 0)
                        {
                            // print avg_gmemUtils.
                            if (verbose)
                            {
                                printf("Device %u: avg mem util at each frequency:", i);
                                for (j = 0; j < numProbFreq; j++)
                                    printf("\t%.2lf", avg_gmemUtils[j]);// from low to high frequency.
                                printf("\n");
                                printf("Device %u: avg device power at each frequency:", i);
                                for (j = 0; j < numProbFreq; j++)
                                    printf("\t%.2lf", avg_gPowers[j]);
                                printf("\n");
                            }
                            optimalFound = false;

                            // Build the performance and power efficiency model to optimize frequency.
                            if (!optimalFound)
                            {
                                // fit the points with a single linear model.
                                linearRegression(numProbRec, x, y, &slope_Opt, &intercept_Opt, &regErr);
                                regErrMin = regErr;
                                turn_Opt = 0;
                                if (verbose)
                                {
                                    printf("Device %u: turn=non, slope=%lf, intercept=%lf, regErr=%lf\n", i, slope_Opt, intercept_Opt, regErr);
                                }
                                // Partition the points and fit the points with two linear models connected by a turning point.
                                for (turn = 2; turn <= numProbFreq - 2; turn++)// "turn" marks how many points are in the 1st model.
                                {
                                    // Partition the points to fit two linear models.
                                    // *1 for lower frequency, and *2 for higher frequency.
                                    idx1 = 0; idx2 = 0;
                                    for (j = 0; j < numProbRec; j++)
                                    {
                                        reminder = j % (2*numProbFreq);
                                        if (reminder < numProbFreq)
                                            ifreq = reminder;
                                        else
                                            ifreq = 2*numProbFreq-1-reminder;
                                        if (ifreq < turn)
                                        {
                                            x1[idx1] = (double)probFreqs[ifreq];// lower frequency.
                                            y1[idx1] = gmemUtils[i][j];
                                            idx1 += 1;
                                        }
                                        else
                                        {
                                            x2[idx2] = (double)probFreqs[ifreq];// higher frequency.
                                            y2[idx2] = gmemUtils[i][j];
                                            idx2 += 1;
                                        }
                                    }
                                    linearRegression(turn*numProbRep, x1, y1, &slope1, &intercept1, &regErr1);
                                    linearRegression((numProbFreq-turn)*numProbRep, x2, y2, &slope2, &intercept2, &regErr2);

                                    if (slope2 != slope1)
                                        freq_cross = (intercept1-intercept2) / (slope2-slope1);
                                    else
                                        freq_cross = -1;
                                    if (freq_cross >= probFreqs[turn-1] && freq_cross <= probFreqs[turn])
                                    {
                                        // this fold-line is valid.
                                        regErr = regErr1 + regErr2;
                                    }
                                    else
                                    {
                                        // re-fit the fold-line and let the cross to happen at probFreqs[turn-1].
                                        foldlineRegression(probFreqs[turn-1], turn*numProbRep, x1, y1, (numProbFreq-turn)*numProbRep, x2, y2, &slope1, &intercept1, &slope2, &intercept2, &regErr);
                                    }

                                    if (verbose)
                                        printf("Device %u: turn=%u, slope1=%lf, intercept1=%lf, slope2=%lf, intercept2=%lf, regErr=%lf. ", i, turn, slope1, intercept1, slope2, intercept2, regErr);

                                    if (slope1 <= slope2)// theoretically impossible case, abandon this partition.
                                    {
                                        if (verbose)
                                            printf("slope1 <= slope2, abandon this partition.\n");
                                    }
                                    else
                                    {
                                        if (regErr < regErrMin)// update this as the optimal model.
                                        {
                                            turn_Opt = turn;
                                            regErrMin = regErr;
                                            slope1_Opt = slope1; intercept1_Opt = intercept1;
                                            slope2_Opt = slope2; intercept2_Opt = intercept2;
                                            if (verbose)
                                                printf("Better model found.\n");
                                        }
                                        else
                                        {
                                            if (verbose)
                                                printf("Larger reg err, not used.\n");
                                        }
                                    }
                                }// end for turn position.

                                // if regression error too large, do not use regression model. Set freq by util.
                                if (regErrMin > numProbRec * regErrThres)
                                {
                                    if (verbose)
                                        printf("All regression err too large, discard model.\n");
                                    skipmodel = true;
                                    // set a high frequency for assurance.
                                    freqBound = (double)maxFreq;// will be bounded by freqCap later.
                                    freqEff = (double)freqAvgEff;
                                }
                                else
                                    skipmodel = false;

                                if (!skipmodel)
                                {
                                    // Estimate power efficiency only at the probed frequencies.
                                    if (turn_Opt == 0)
                                    {
                                        for (j = 0; j < numProbFreq; j++)
                                        {
                                            if (slope_Opt > 0)// assume performance correlates to gmemutil.
                                                modelPerf[j] = slope_Opt*(double)probFreqs[j]+intercept_Opt;
                                            else // assume the lowest frequency's performance is maximal.
                                                modelPerf[j] = slope_Opt*(double)probFreqs[0]+intercept_Opt;
                                        }
                                    }
                                    else // fold-line model.
                                    {
                                        // slope1_Opt for lower frequency, and slope2_Opt for higher.
                                        if (slope1_Opt > 0 && slope2_Opt > 0)
                                        {
                                            for (j = 0; j < numProbFreq; j++)
                                            {
                                                if (j >= turn_Opt)
                                                {
                                                    // model for higher frequency.
                                                    modelPerf[j] = slope2_Opt*(double)probFreqs[j]+intercept2_Opt;
                                                }
                                                else
                                                {
                                                    // model for lower frequency.
                                                    modelPerf[j] = slope1_Opt*(double)probFreqs[j]+intercept1_Opt;
                                                }
                                            }
                                        }
                                        else if (slope2_Opt <= 0 && slope1_Opt > 0)// maximum is in the middle.
                                        {
                                            for (j = 0; j < numProbFreq; j++)
                                            {
                                                if (j < turn_Opt)
                                                {
                                                    // model for lower frequency.
                                                    modelPerf[j] = slope1_Opt*(double)probFreqs[j]+intercept1_Opt;
                                                }
                                                else
                                                {
                                                    // use the estimated performance at cross.
                                                    freq_cross = (intercept1_Opt-intercept2_Opt) / (slope2_Opt-slope1_Opt);
                                                    modelPerf[j] = (slope2_Opt*intercept1_Opt-slope1_Opt*intercept2_Opt) / (slope2_Opt-slope1_Opt);
                                                }
                                            }
                                        }
                                        else // slope1_Opt <= 0.
                                        {
                                            for (j = 0; j < numProbFreq; j++)
                                            {
                                                // estimate performance using the lowest frequency.
                                                modelPerf[j] = slope1_Opt*(double)probFreqs[0]+intercept1_Opt;
                                            }
                                        }
                                    }//end if model with turing point.

                                    // calculate the power efficiency.
                                    for (j = 0; j < numProbFreq; j++)
                                    {
                                        powerEffici[j] = modelPerf[j] / avg_gPowers[j];
                                    }
                                    if (verbose)
                                    {
                                        printf("Device %u: modeled performance:", i);
                                        for (j = 0; j < numProbFreq; j++)
                                            printf("\t%lf", modelPerf[j]);// print from low to high frequency.
                                        printf("\n");
                                        printf("Device %u: power efficiency:", i);
                                        for (j = 0; j < numProbFreq; j++)
                                            printf("\t%lf", powerEffici[j]);// print from low to high frequency.
                                        printf("\n");
                                    }

                                    // find the most power efficient frequency.
                                    mostEffici = powerEffici[0];
                                    mostEfficiFreq = probFreqs[0];
                                    for (j = 1; j < numProbFreq; j++)
                                    {
                                        if (powerEffici[j] > mostEffici)
                                        {
                                            mostEffici = powerEffici[j];
                                            mostEfficiFreq = probFreqs[j];
                                        }
                                    }
                                    freqEff = (double)mostEfficiFreq;
                                    if (verbose)
                                        printf("Device %u: max efficiency %lf at frequency %d MHz.\n", i, mostEffici, mostEfficiFreq);

                                    // calculate critical frequency bounded by performance constraint using gmem util model.
                                    if (turn_Opt == 0) // if a single linear model is optimal.
                                    {
                                        if (slope_Opt > 0)
                                            freq_perfBound = (perfThres*(slope_Opt*(double)maxFreq+intercept_Opt) - intercept_Opt) / slope_Opt;
                                        else // lower frequency is better.
                                            freq_perfBound = (double)probFreqs[0];
                                        if (verbose)
                                            printf("Performance estimated by single linear model.\n");
                                    }
                                    else // if fold-line model.
                                    {
                                        if (slope1_Opt > 0)
                                        {
                                            if (slope2_Opt > 0)
                                            {
                                                criticalPerf = perfThres*(slope2_Opt*(double)maxFreq+intercept2_Opt);
                                                freq_perfBound = (criticalPerf - intercept2_Opt) / slope2_Opt;
                                                freq_cross = (intercept1_Opt-intercept2_Opt) / (slope2_Opt-slope1_Opt);
                                                if (freq_perfBound <= freq_cross)
                                                {
                                                    // should use low-freq-model instead.
                                                    freq_perfBound = (criticalPerf - intercept1_Opt) / slope1_Opt;
                                                    if (verbose)
                                                        printf("Performance assurance satisfied at low-segment.\n");
                                                }
                                                else
                                                {
                                                    if (verbose)
                                                        printf("Performance assurance satisfied at high-segment.\n");
                                                }
                                            }
                                            else if (slope2_Opt <= 0)
                                            {
                                                freq_cross = (intercept1_Opt-intercept2_Opt) / (slope2_Opt-slope1_Opt);
                                                criticalPerf = perfThres*(slope1_Opt*freq_cross+intercept1_Opt);
                                                if (verbose)
                                                    printf("Performance saturation predicted at %.lf MHz.\n", freq_cross);
                                                freq_perfBound = (criticalPerf - intercept1_Opt) / slope1_Opt;
                                                if (verbose)
                                                    printf("Performance assurance satisfied at low-segment.\n");
                                            }
                                        }// end if slope1_Opt>0.
                                        else
                                        {
                                            if (verbose)
                                                printf("Performance saturation predicted at %d MHz.\n", probFreqs[0]);
                                            freq_perfBound = (double)probFreqs[0];
                                        }
                                    }// end if fold-line model.
                                    if (verbose)
                                        printf("Device %u: performance assurance achieved at %.2lf MHz.\n", i, freq_perfBound);

                                    freqBound = freq_perfBound;
                                }// end if !skipmodel.
                            }// end if !optimalFound.
                        }// end if sumy > 0.
                        else
                        {
                            if (verbose)
                                printf("Device %u: mem bw not used, will set frequency by util.\n", i);
                            freqBound = (double)maxFreq;// will be bounded by freqCap later.
                            freqEff = (double)freqAvgEff;// on A100, gmemutil may be always 0 for a few apps.
                        }
                    } // end if useRegression.
                    else // use the lowest frequency whose gmemUtil is maximal.
                    {
                        max_gmem = avg_gmemUtils[0];
                        max_gmem_freq = probFreqs[0];
                        // get the max_gmem value;
                        for (j = 1; j < numProbFreq; j++)
                        {
                            if (avg_gmemUtils[j] > max_gmem)
                            {
                                max_gmem = avg_gmemUtils[j];
                            }
                        }
                        // get the freq of max_gmem.
                        for (j = 0; j < numProbFreq; j++)
                        {
                            if (avg_gmemUtils[j] >= max_gmem*0.99)
                            {
                                max_gmem_freq = probFreqs[j];
                                break;
                            }
                        }
                        freqBound = (double)max_gmem_freq;
                        freqEff = (double)freqAvgEff;
                    }

                    if (useFreqCap)
                    {
                        freqPerf = min(freqBound, freqCap[i]);
                        if (verbose && freqBound > freqCap[i])
                            printf("Device %u: set frequency %.1f capped by gpu util.\n", i, freqCap[i]);
                    }
                    else
                    {
                        freqPerf = freqBound;
                    }
                    freqOpt = max(freqPerf, freqEff);
                    if (verbose && freqPerf >= freqEff)
                        printf("Device %u, selecting the performance-assured frequency.\n", i);
                    if (verbose && freqPerf < freqEff)
                        printf("Device %u, selecting the most power efficient frequency.\n", i);

                    // set optimized freq by looking through the available freq list.
                    freqOpt = max(freqOpt, (double)minSetFreq);
                    freqOpt = min(freqOpt, (double)maxFreq);
                    for (iAvail = numAvailableFreqs-1; iAvail >= 0; iAvail--)
                    {
                        if (availableFreqs[iAvail] < freqOpt)
                        {
                            if (iAvail < numAvailableFreqs-1)
                                optimizedFreqs[i] = availableFreqs[iAvail+1];
                            else
                                optimizedFreqs[i] = availableFreqs[iAvail];
                            break;
                        }
                    }
                }// end for device index.

                if (verbose)
                {
                    printf("Optimized frequencies:");
                    for (i = 0; i < device_count; i++)
                        printf("\t%d", optimizedFreqs[i]);
                    printf("\n");
                }
            }// end if just finished probing.
        }// end if Assure.

        // wait until the next loopDelay millisecond if not already reached.
        gettimeofday(&endtime, NULL);
        duration = (endtime.tv_sec - starttime.tv_sec) * 1000000 + endtime.tv_usec - starttime.tv_usec;
        printf("%lu\n", duration);
        if (duration < loopDelay*1000)// loopDelay is in milliseconds.
        {
            usleep(loopDelay*1000-duration);
            addTime = loopDelay*1000;
        }
        else
            addTime = duration;

        if (strcmp(argv[1], "mod") == 0 && strcmp(freqsetAlg, "Assure") == 0)
        {
            // determine whether or not enter the probing phase.
            lastprobPhase = probPhase;
            if (accumuTime >= probDelay*1000000)// probDelay is in seconds.
            {
                // every probDelay seconds, check if process exist.
                // If so (sum_gutil >=1), start the probing phase to get util values at a range of frequencies.
                sum_gutil = 0;
                for (i = 0; i < device_count; i++)
                {
                    sum_gutil += gutil_moving_avg[i];
                }
                if (sum_gutil >= 1)// sum_gutil is double type.
                {
                    if (verbose)
                    {
                        printf("Probing phase start at ");
                        time(&t);
                        lt = localtime(&t);
                        printf("%d-%d-%d %d:%d:%d\n" ,lt->tm_year+1900, lt->tm_mon+1, lt->tm_mday, lt->tm_hour, lt->tm_min, lt->tm_sec);
                    }
                    probPhase = numProbRec;// probPhase traces the execution of the probing phase.
                }
                else
                {
                    probPhase = -2;
                    if (verbose)
                        printf("Negligible avg util. Probing omitted.\n");
                }

                accumuTime = 0;// reset accumuTime.
            }
            else
            {
                if (probPhase > -1)
                    accumuTime = 0; // only accumulate time after probing phase.
                else
                    accumuTime += addTime;
                if (probPhase > -99) // use a low limit to prevent overflow.
                    probPhase -= 1;
            }
        }// end if Assure.
        initialLoop = false;
    }// end of main while loop.

    // Reset GPU clocks before terminate.
    printf("Reset GPU frequency for: ");
    for (i = 0; i < device_count; i++)
    {
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (NVML_SUCCESS != result)
        { 
            printf("Failed to get handle for GPU %u: %s\n", i, nvmlErrorString(result));
            goto Error;
        }

        result = nvmlDeviceResetGpuLockedClocks(device);
        if (NVML_ERROR_NO_PERMISSION == result)
            printf("\t\t Need root privileges: %s\n", nvmlErrorString(result));
        else if (NVML_ERROR_NOT_SUPPORTED == result)
            printf("\t\t Operation not supported.\n");
        else if (NVML_SUCCESS != result)
        {
            printf("\t\t Failed to reset locked frequency for GPU %u: %s\n", i, nvmlErrorString(result));
            goto Error;
        } 

        result = nvmlDeviceResetApplicationsClocks(device);
        if (NVML_ERROR_NO_PERMISSION == result)
            printf("\t\t Need root privileges: %s\n", nvmlErrorString(result));
        else if (NVML_ERROR_NOT_SUPPORTED == result)
            printf("\t\t Operation not supported.\n");
        else if (NVML_SUCCESS != result)
        {
            printf("\t\t Failed to reset application frequency for GPU %u: %s\n", i, nvmlErrorString(result));
            goto Error;
        } 
        else if (NVML_SUCCESS == result)
        {
            printf("device %u. ", i);
        }
    }
    printf("\n");

    // Terminate.
    free(probFreqs);
    free(optimizedFreqs);
    for (i = 0; i < device_count; i++)
    {
        free(gpuUtils[i]);
        free(gpuUtils_sq[i]);
        free(gmemUtils[i]);
        free(gPowers[i]);
    }
    free(gpuUtils);
    free(gpuUtils_sq);
    free(gmemUtils);
    free(gPowers);
    free(modelPerf);
    free(allModelPerf);
    free(allModelPower);
    free(allPowerEffici);
    free(powerEffici);
    free(gutil_moving_avg);
    free(gutil_moving_sqsum);
    free(gutil_moving_std);
    free(avg_gmemUtils);
    free(freqCap);
    free(x); free(x1); free(x2); free(x3);
    free(y); free(y1); free(y2); free(y3);
    result = nvmlShutdown();
    if (NVML_SUCCESS != result)
        printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
    printf("GPU freqset tool terminated.\n");
    return 0;

Error:
    result = nvmlShutdown();
    if (NVML_SUCCESS != result)
        printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
    return 1;
}
// End of file dvfs.c
