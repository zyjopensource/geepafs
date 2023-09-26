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

 * GPU metrics reading and frequency tuning latency measurement.
 * If changeFreq is true, it sets GPU frequency to oscillate between two values to measure latency.
 * Use "make" to compile.
 * Run by "sudo ./measure_latency -1" to measure all GPUs; change to a number other than -1 measures the specified GPU.
 * Use Ctrl-C to stop. Frequency is reset automatically at stop.
 * Bool constants printDateTime, getUtilization, getFrequency, getPower, changeFreq can be adjusted.
 */
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <nvml.h>
#include <sys/time.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <time.h>

 // Please select one of the following MACHINE choices by editing the "#define" lines.:
 // ("v100-maxq" is V100 GPU with TDP 163 W; "v100-300w" is V100 GPU with TDP 300 W; "a100-insp" is A100 GPU with TDP 400 W.)

//#define MACHINE "v100-maxq"
#define MACHINE "v100-300w"
//#define MACHINE "a100-insp"

static volatile int keepRunning = 1;
void intHandler(int dummy) // function used for the interruptable loop.
{
    keepRunning = 0;
}

int main(int argc, char* argv[])
{
    unsigned int minSetFreq;
    unsigned int maxFreq;
    unsigned int setMemFreq;

    // Run "nvidia-smi -q -d SUPPORTED_CLOCKS" to get available frequencies and update the following parameters if needed.
    if (strcmp(MACHINE, "v100-maxq") == 0)
    {
        minSetFreq = 855; maxFreq = 1440; setMemFreq = 810;
    }
    else if (strcmp(MACHINE, "v100-300w") == 0)
    {
        minSetFreq = 952; maxFreq = 1530; setMemFreq = 877;
    }
    else if (strcmp(MACHINE, "a100-insp") == 0)
    {
        minSetFreq = 1110; maxFreq = 1410; setMemFreq = 1593;
    }

    printf("Argument: %s\n",argv[1]);
    int argument = atoi(argv[1]);// gpu ID. -1 means all gpu.
    const int loopDelay = 200;// interval of each loop in milliseconds.
    const bool onlySetAppFreq = true;// default true. true - nvmlDeviceSetApplicationsClocks(); false - nvmlDeviceSetGpuLockedClocks().
    const bool printDateTime = false;
    const bool getUtilization = true;
    const bool getFrequency = false;
    const bool getPower = true;
    const bool changeFreq = false;

    nvmlReturn_t result;
    nvmlDevice_t device;
    nvmlUtilization_t util;
    nvmlClockType_t freq;
    unsigned int power, device_count, i, setFreq, iterflag;// warning: unsigned int cannot loop from high to low.
    struct timeval starttime, endtime;
    long unsigned int duration;
    time_t t;
    struct tm * lt;

    // Initialize.
    printf("MACHINE %s\n", MACHINE);
    printf("GPU freqset latency measurement start..\n");
    printf("Latency is shown in the rightmost column (unit: microsecond).\n");
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

    // Reset all GPU clocks.
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
    iterflag = 0;
    signal(SIGINT, intHandler);
    printf("Main loop start..\n");
    while (keepRunning) // press ctrl+c can break this loop.
    {
        gettimeofday(&starttime, NULL);
        if (printDateTime)
        {
            time(&t);
            lt = localtime(&t);
            printf("%d-%d-%d %d:%d:%d, " ,lt->tm_year+1900, lt->tm_mon+1, lt->tm_mday, lt->tm_hour, lt->tm_min, lt->tm_sec);
        }
        // loop through each GPU.
        for (i = 0; i < device_count; i++)
        {
            if ((argument == -1) || (argument == i))// this will skip other GPUs.
            {
                result = nvmlDeviceGetHandleByIndex(i, &device);
                if (NVML_SUCCESS != result)
                { 
                    printf("Failed to get handle for GPU %u: %s\n", i, nvmlErrorString(result));
                    goto Error;
                }

                if (getUtilization)
                {
                    // get gpu utilization rate (including gmem bandwidth util).
                    result = nvmlDeviceGetUtilizationRates(device, &util);
                    if (NVML_SUCCESS != result)
                    { 
                        printf("Failed to get utilization rate for GPU %u: %s\n", i, nvmlErrorString(result));
                        goto Error;
                    }
                    else
                        printf("%u, %u, ", util.gpu, util.memory);
                }

                if (getFrequency)
                {
                    // get gpu frequency.
                    result = nvmlDeviceGetClockInfo(device, 1, &freq);// 1 refers to SM domain.
                    if (NVML_SUCCESS != result)
                    { 
                        printf("Failed to get clock frequency for GPU %u: %s\n", i, nvmlErrorString(result));
                        goto Error;
                    }
                    else
                        printf("%u, ", freq);
                }

                if (getPower)
                {
                    // get gpu power usage.
                    result = nvmlDeviceGetPowerUsage(device, &power);
                    if (NVML_SUCCESS != result)
                    {
                        printf("Failed to get power usage for GPU %u: %s\n", i, nvmlErrorString(result));
                        goto Error;
                    }
                    else
                        printf("%u, ", power);
                }

                if (changeFreq)
                {
                    // set freq to simply oscillate between two values.
                    if (iterflag == 0)
                        setFreq = minSetFreq;
                    else
                        setFreq = maxFreq;

                    // Execute frequency set.
                    if (onlySetAppFreq)
                    {
                        result = nvmlDeviceSetApplicationsClocks(device, setMemFreq, setFreq);
                    }
                    else
                    {
                        result = nvmlDeviceSetGpuLockedClocks(device, setFreq, setFreq);
                    }

                    if (NVML_ERROR_NO_PERMISSION == result)
                        printf("\t\t Error: Need root privileges: %s\n", nvmlErrorString(result));
                    else if (NVML_ERROR_NOT_SUPPORTED == result)
                        printf("\t\t Operation not supported.\n");
                    else if (NVML_SUCCESS != result)
                    {
                        printf("\t\t Failed to set frequency for GPU %u: %s\n", i, nvmlErrorString(result));
                        goto Error;
                    } 
                    else if (NVML_SUCCESS == result)
                    {
                        printf("%u, ", setFreq);
                    }
                }
            }// end if not the GPU to be measured.
        }// loop all GPU ends.

        // wait until the next loopDelay millisecond if not already reached.
        gettimeofday(&endtime, NULL);
        duration = (endtime.tv_sec - starttime.tv_sec) * 1000000 + endtime.tv_usec - starttime.tv_usec;
        printf("%lu\n", duration);
        if (duration < loopDelay*1000)// loopDelay is in milliseconds.
        {
            usleep(loopDelay*1000-duration);
        }
        // change iterflag.
        if (iterflag == 0)
            iterflag = 1;
        else
            iterflag = 0;
    }// end of main while loop.

    // Reset all GPU clocks before terminate.
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
// End of file measure_latency.c