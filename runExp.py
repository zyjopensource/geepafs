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
Scripts for running GPU frequency scaling experiments.
This code will launch the compiled dvfs.c code and compiled benchmarks as subprocesses.
The dvfs.c and the benchmarks need to be compiled before running this code.

Run this code using "sudo python3 runExp.py"
Root privileges are necessary in applying frequency tuning.
Results are saved to the ./output/ folder.
'''
import time
import subprocess
import os
import signal

def main():
    # policy can be one of: Assure, MaxFreq, EfficientFix, UtilizScale, NVboost.
    # iterations set the number of times each benchmark is launched.
    # suffix is a suffix to the output file name.
    # assurance variable is only effective when policy=='Assure'.
    allApps(policy='Assure', iterations=2, suffix='_new', startdvfs=True, assurance=90)

def getAppCmd(app, gpu):
    cudaSDKdir = './cuda_samples/benchmarks'

    if app=='cudaTensorCoreGemm':
        appcmd = './cudaTensorCoreGemm'
        workdir = '%s/cudaTensorCoreGemm' % cudaSDKdir
    elif app=='bandwidthTest':
        appcmd = './bandwidthTest --dtod --device=0'
        workdir = '%s/bandwidthTest' % cudaSDKdir
    elif app=='sortingNetworks':
        appcmd = './sortingNetworks'
        workdir = '%s/sortingNetworks' % cudaSDKdir
    elif app=='transpose':
        appcmd = './transpose'
        workdir = '%s/transpose' % cudaSDKdir
    elif app=='BlackScholes':
        appcmd = './BlackScholes'
        workdir = '%s/BlackScholes' % cudaSDKdir
    elif app=='fastWalshTransform':
        appcmd = './fastWalshTransform'
        workdir = '%s/fastWalshTransform' % cudaSDKdir
    elif app=='reductionMultiBlockCG':
        appcmd = './reductionMultiBlockCG'
        workdir = '%s/reductionMultiBlockCG' % cudaSDKdir
    elif app=='convolutionFFT2D':
        appcmd = './convolutionFFT2D'
        workdir = '%s/convolutionFFT2D' % cudaSDKdir
    return appcmd, workdir

def allApps(policy, iterations, suffix, startdvfs, assurance):
    import random
    if policy == 'Assure':
        outputfile = 'allApps_%s_p%d_%diter%s.out' % (policy, assurance, iterations, suffix)
    else:
        outputfile = 'allApps_%s_%diter%s.out' % (policy, iterations, suffix)

    apps = ['convolutionFFT2D','reductionMultiBlockCG','fastWalshTransform','BlackScholes','transpose','sortingNetworks','bandwidthTest','cudaTensorCoreGemm']

    print('Starting..')
    outfile = open('output/%s' % outputfile,'w')

    if startdvfs:
        if policy == 'Assure':
            dvfscmd = 'sudo ./dvfs mod %s p%d > output/dvfs_%s_p%d%s.out' % (policy, assurance, policy, assurance, suffix)
        else:
            dvfscmd = 'sudo ./dvfs mod %s > output/dvfs_%s%s.out' % (policy, policy, suffix)
        dvfs = subprocess.Popen(dvfscmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, preexec_fn=os.setsid)# preexec_fn is necessary.
        print('dvfs process started.')

    # this one is a warm-up run. Should not be used in results.
    print('Start warm-up run.')
    appcmd, workdir = getAppCmd('cudaTensorCoreGemm', gpu=1)
    apprun = subprocess.Popen(appcmd, stdout=outfile, stderr=outfile, shell=True, cwd=workdir)
    apprun.wait()
    outfile.flush()

    # experiment start.
    for iteration in range(iterations):
        print('Iteration %d:' % iteration)
        outfile.write('=====================================\n')
        outfile.write('Iteration %d:\n' % iteration)
        random.shuffle(apps)# shuffle apps to mitigate potential impact from a prior run.
        for app in apps:
            time.sleep(20)
            print(app)
            outfile.write('Application name: %s\n' % app)
            outfile.flush()
            appcmd, workdir = getAppCmd(app, gpu=1)
            apprun = subprocess.Popen(appcmd, stdout=outfile, stderr=outfile, shell=True, cwd=workdir) # stderr needs to be added, otherwise the datetime is not included.
            apprun.wait()
            outfile.flush()

    # experiment finished.
    time.sleep(10)
    if startdvfs:
        os.killpg(os.getpgid(dvfs.pid), signal.SIGTERM)
        print('dvfs process killed.')
    outfile.close()
    print('Finished.')

if __name__ == '__main__':
    main()
