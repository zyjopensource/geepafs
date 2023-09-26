'''
MIT License

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

###
Scripts for post-processing experimental results.
This code takes .out files as input, and outputs .csv files.
'''
import pandas as pd
from datetime import datetime

def main():
    parseAllApps(resultfile='allApps_Assure_p90_2iter_demo.out', gpufile='dvfs_Assure_p90_demo.out', outfile='processed_Assure_p90_demo.csv')

def appResultLine(app):
    resultLine = 'noResultLine'
    if app == 'cudaTensorCoreGemm':
        resultLine = 'TFLOPS'
    elif app == 'bandwidthTest':
        resultLine = '   32000000'
    elif app == 'sortingNetworks':
        resultLine = 'sortingNetworks-bitonic'
    elif app == 'BlackScholes':
        resultLine = 'BlackScholes, Throughput'
    elif app == 'fastWalshTransform':
        resultLine = 'GPU time'
    elif app == 'convolutionFFT2D':
        resultLine = 'Result'
    return resultLine

def appEndLine(app):
    if app == 'cudaTensorCoreGemm':
        endLine = 'cudaTensorCoreGemm test end'
    elif app == 'bandwidthTest':
        endLine = 'Result = PASS'
    elif app == 'sortingNetworks':
        endLine = 'Shutting down'
    elif app == 'transpose':
        endLine = 'Test passed'
    elif app == 'BlackScholes':
        endLine = 'Shutdown done'
    elif app == 'fastWalshTransform':
        endLine = 'Shutting down'
    elif app == 'reductionMultiBlockCG':
        endLine = 'CPU result'
    elif app == 'convolutionFFT2D':
        endLine = 'Test passed'
    return endLine

def appTimeLine(app, line):
    linetime = datetime.strptime(line[:-1], '%Y-%m-%d %H:%M:%S')
    return linetime

def appPerfTime_fromResultLine(app, line):
    performance = -2
    exetime = -2
    if app == 'cudaTensorCoreGemm':
        performance = float(line.split()[1])
    elif app == 'bandwidthTest':
        performance = float(line.split()[1])
    elif app == 'sortingNetworks':
        performance = float(line.split()[3])
    elif app == 'BlackScholes':
        performance = float(line.split()[3])
    elif app == 'fastWalshTransform':
        performance = float(line.split()[5]) # Gops
    elif app == 'convolutionFFT2D':
        performance = float(line.split()[1]) # MPix/s
    return performance, exetime

def appPerf_fromEndLine(app, performance, exetime):
    performance = performance # read existing value.
    if app == 'transpose':
        performance = 8210000/exetime # iterations per second.
    elif app == 'reductionMultiBlockCG':
        performance = 474000/exetime # iterations per second.
    return performance

def getGpuPowerEffici_dataframe(app, gpupowerlines, numGPU, starttime, endtime, startidx, detectGPU, gpuList):
    '''
    Read the GPU metrics from file, use dataframe to process data and get the averaged metric values.
    May set gpu list by gpuList. Default is [].
    detectGPU: Auto-detect used GPU index, report if not matching used GPU number: 'numGPU'. Avg GPU util > 1 treated as a used GPU.
    readidx: file start position, used for quick jump.
    '''
    cols = ['time']
    Tform = '%Y-%m-%d %H:%M:%S'
    for i in range(8):# there are at most 8 GPUs in our server.
        cols = cols + ['gutil_%d' % i, 'gmemutil_%d' % i, 'gpower_%d' % i, 'gfreq_%d' % i, 'gfreqset_%d' % i]
    cols.append('delay')
    tempfile = open('temp_getGpuPowerEffici_dataframe.csv', 'w') # a temporary file, can be deleted later.
    jump, readidx = 0, startidx
    while 1:
        readidx += jump
        gpupowerline = gpupowerlines[readidx]
        if not gpupowerline.startswith('202'):# 202x year.
            jump = 1
            continue
        powerT = datetime.strptime(gpupowerline.split(',')[0], Tform)
        if powerT < starttime:
            diff = (starttime - powerT).seconds
            jump = max(int(diff * 0.5), 1)
        else:
            if powerT <= endtime:
                tempfile.write(gpupowerline)
                jump = 1
            else:
                break
    tempfile.close()
    df = pd.read_csv('temp_getGpuPowerEffici_dataframe.csv', sep=', ', names=cols)
    avg = df.mean()
    if detectGPU:
        usedGPU = [i for i in range(8) if avg['gutil_%d' % i]>1]
        if len(usedGPU) != numGPU:
            print('App %s, %d used GPU detected, not matching the preset %d GPU.' % (app, len(usedGPU), numGPU))
    else:
        if len(gpuList) == 0:
            usedGPU = list(range(numGPU))
        else:
            usedGPU = gpuList
    gutilValue = sum([avg['gutil_%d' % i] for i in usedGPU])/len(usedGPU)
    gmemutilValue = sum([avg['gmemutil_%d' % i] for i in usedGPU])/len(usedGPU)
    gpupowerValue = sum([avg['gpower_%d' % i]/1000 for i in usedGPU])
    gpufreqValue = sum([avg['gfreq_%d' % i] for i in usedGPU])/len(usedGPU)
    return gpupowerValue, gutilValue, gmemutilValue, gpufreqValue, readidx

def parseAllApps(resultfile, gpufile, outfile):
    '''
    Read and parse all apps.
    In a server with multiple GPUs, we assume the GPUs not running our benchmark
    applications are not utilized during experiments. Then, 'detectGPU' if true
    will select the used GPU based on average GPU utilization.
    Or, you may also put the GPU index that runs the benchmarks into 'gpuList'.
    '''
    folder = '.'
    detectGPU = True
    gpuList = []
    
    result = open('%s/output/%s' % (folder, outfile), 'w')
    appout = open('%s/output/%s' % (folder, resultfile), 'r', encoding='utf-8')
    with open('%s/output/%s' % (folder, gpufile), 'r') as gpuf:
        gpupowerlines = gpuf.readlines()
    headline = 'iter,app,performance,gpupower(W),power-efficiency,exetime(s),start,end,gpuUtil,gmemUtil,frequency'
    result.write(headline+'\n')
    app, resultLine, endLine = 'noApp', 'noLine', 'noLine'
    iteration, thistime, lasttime, performance, readidx = -1, -1, -1, -1, 1
    for line in appout:
        if line.startswith('Iteration') and line.endswith(':\n'):
            iteration = int(line.split()[1][:-1])
            print('Iteration: %d' % iteration)
        if iteration >= 0:# skip the warm-up run.
            if line.startswith('Application name:') and len(line.split())==3:
                app = line.split()[2]
                resultLine = appResultLine(app)
                endLine = appEndLine(app)
            if line.startswith('202') or line.startswith('[202'):# 202x year.
                lasttime = thistime
                thistime = appTimeLine(app, line)
            if line.startswith(resultLine):
                performance, exetime = appPerfTime_fromResultLine(app, line)
            if line.startswith(endLine) or line.endswith(endLine):# sometimes the time line and result line change order.
                exetime = (thistime - lasttime).seconds
                performance = appPerf_fromEndLine(app, performance, exetime)
                numGPU = 1 # all of the cuda samples use only 1 GPU card.
                gpupowerValue, gutilValue, gmemutilValue, gpufreqValue, readidx = getGpuPowerEffici_dataframe(app, 
                    gpupowerlines, numGPU, lasttime, thistime, startidx=readidx, detectGPU=detectGPU, gpuList=gpuList)
                gpuEff = performance / gpupowerValue # power efficiency of GPU.
                start = lasttime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]# microseconds are removed.
                end = thistime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                dataline = '%d,%s,%.3f,%.1f,%.3f,%.2f,%s,%s,%.2f,%.2f,%.f' % (iteration, app, performance, gpupowerValue, gpuEff, exetime, start, end, gutilValue, gmemutilValue, gpufreqValue)
                result.write(dataline + '\n')
    appout.close()
    result.close()
    # further calculate the average of all runs.
    df = pd.read_csv('%s/output/%s' % (folder, outfile))
    avgIter = df.groupby(df['app']).mean()
    avgIter['exetime_std'] = df.groupby(df['app']).std()['exetime(s)'] # add standard deviation.
    avgIter['gpupower_std'] = df.groupby(df['app']).std()['gpupower(W)']
    avgIter.to_csv('%s/output/avgIter_%s' % (folder, outfile))

if __name__ == '__main__':
    main()