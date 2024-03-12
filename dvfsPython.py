"""
MIT License

Copyright (c) 2024 Yijia Zhang

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

 * GPU Energy-Efficient and Performance-Assured Frequency Scaling (GEEPAFS) policy python version.
 * This program reads NVIDIA GPU metrics using DCGM and tunes the GPU frequency using NVML.

 * Select the correct GPU type by editing the "MACHINE =" line.
 * Run with default settings by the command "sudo python dvfsPython.py Assure 90". The last number 90 represents the performance constraint 90%.
 * Root privileges are necessary in applying frequency tuning.
 * Require install DCGM and pynvml. By default, the DcgmReader class of DCGM is installed in /usr/local/dcgm/bindings/python3/. If installed to another place, that path should be added using sys.path.append() function like our example.
 * Python 3 is preferred, and Python 2 may not work. Note: executing with "sudo python" may lead to a different version than "python". You can use the full path "sudo /path/to/your/python" to avoid that issue.
 * This program runs endlessly. Press ctrl-c to stop. Frequency is reset automatically at stop.

 * "verbose" variable can be initialized as true to output more details for debugging.
 * If running in linux screen utility, then output to a file is necessary to prevent buffer jam.
 * Some python codes here are not optimized in order to match the C version.

"""
import time
from datetime import datetime
import sys
sys.path.append(r"/usr/local/dcgm/bindings/python3/") # update this path if your DCGM is installed in a different place.
import DcgmReader # this class is inside the path above.
import dcgm_fields
import pynvml
from collections import deque
import numpy as np
import signal

# Please select one of the following MACHINE choices by editing the variable. ("v100-maxq" is V100 GPU with TDP 163 W; "v100-300w" is V100 GPU with TDP 300 W; "a100-insp" is A100 GPU with TDP 400 W.)
MACHINE = "v100-maxq"
# Other GPU types are not directly supported so far. To use this code for another GPU type, constants minSetFreq, freqAvgEff, maxFreq, setMemFreq, probFreqs, and the function getAvailableFreqs() need to be updated.

selectGPU = -1 # select a GPU ID to apply the policy. -1 means all GPUs.
policy = sys.argv[1] #"Assure" # Assure, NVboost, MaxFreq, EfficientFix, UtilizScale.
if policy == "Assure":
    perfThres = float(sys.argv[2])/100 # =0.90 # Performance threshold in Assure.
verbose = 0
if MACHINE == "v100-300w":
    minSetFreq = 952 # The lower bound for setting frequency. Used in multiple policies.
    freqAvgEff = 952 # the globally most power efficient frequency based on experiments from many apps.
    maxFreq = 1530 # max freq supported.
    setMemFreq = 877 # the only available memory freq value on this machine.
    probFreqs = [952,1147,1335,1530] # frequency values for probing.
elif MACHINE == "v100-maxq":
    minSetFreq = 720 # The lower bound for setting frequency. Used in multiple policies.
    freqAvgEff = 855 # the globally most power efficient frequency based on experiments from many apps.
    maxFreq = 1440 # max freq supported.
    setMemFreq = 810 # the only available memory freq value on this machine.
    probFreqs = [720,855,982,1117,1245,1440] # frequency values for probing.
elif MACHINE == "a100-insp":
    minSetFreq = 1110 # The lower bound for setting frequency. Used in multiple policies.
    freqAvgEff = 1110 # the globally most power efficient frequency based on experiments from many apps.
    maxFreq = 1410 # max freq supported.
    setMemFreq = 1593 # the only available memory freq value on this machine.
    probFreqs = [1110,1215,1320,1410] # frequency values for probing.
else:
    print("Please set correct MACHINE parameter or add your device.")
    sys.exit(0)
numProbFreq = len(probFreqs)
loopDelay = 0.3 # delay for the while loop, in seconds.
update_frequency = 50000 # internal update frequency of DCGM, in microseconds.
probDelay = 18 # interval between two probing phase in seconds.
numProbRep = 2 # reptition of each frequency point in the probing phase.
useRegression = 1 # default true(1). do the 1-fold foldline regression.
regErrThres = 0.01 # average regression error threshold per point, beyond which regression model is discarded.
calcAllEffici = 0 # default false(0). Whether calculate the power efficiency for all frequencies or only calculate at the probing frequencies. Calculate for all frequencies will fit a 3rd-order polynomial freq-power model.
movingAvg_windowSize = 16 # size (steps) for the moving avg window.
numProbRec = numProbFreq * numProbRep
useFreqCap = 1 # default true(1). whether set an upper bound by utilization.
gmembwLowThres = 0.03 # if gpu mem bw lower than this, do not apply Assure policy.
probInterval = 20 # interval used in baseline policies, in cycles.
setAppFreq = True # default true. true - nvmlDeviceSetApplicationsClocks(); false - nvmlDeviceSetGpuLockedClocks().
skipSetFreq = 0 # default false(0). true will skip setting freq to the optimized value, and this is only used to measure the cost of the probing phase.
customFields = [dcgm_fields.DCGM_FI_PROF_GR_ENGINE_ACTIVE,
                dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE, 
                dcgm_fields.DCGM_FI_DEV_SM_CLOCK, 
                dcgm_fields.DCGM_FI_DEV_POWER_USAGE]

class MovingAverage:
    def __init__(self, size):
        self.size = size
        self.queue = deque()
        self.sum = 0
    def addData(self, value):
        if len(self.queue) == self.size:
            self.sum -= self.queue.popleft()
        self.queue.append(value)
        self.sum += value
    def getMovingAverage(self):
        if not self.queue:
            return 0
        return self.sum / len(self.queue)
    def getMovingSum(self):
        if not self.queue:
            return 0
        return self.sum

def linearRegression(X, Y):
    if len(X) != len(Y):
        print("Error: X Y dimensions do not match in linearRegression.")
    X = np.array(X)
    Y = np.array(Y)
    slope, intercept = np.polyfit(X, Y, 1)
    regErr = np.sqrt(np.sum((Y - (slope * X + intercept)) ** 2) / len(X))
    return slope, intercept, regErr

def foldlineRegression(xc, X1, Y1, X2, Y2):
    """
    Fold-line regression with the assumption that the fold-point's x position is at xc.
    The formula are derived analytically by minimizing square error.
    """
    if len(X1) != len(Y1):
        print("Error: X1 Y1 dimensions do not match in foldlineRegression.")
    if len(X2) != len(Y2):
        print("Error: X2 Y2 dimensions do not match in foldlineRegression.")
    err = 0
    sum1_x = sum1_y = sum1_xsq = sum1_xy = 0
    sum2_x = sum2_y = sum2_xsq = sum2_xy = 0
    num1, num2 = len(X1), len(X2)
    n = num1 + num2
    # Summation for first set of points
    for i in range(num1):
        sum1_x += X1[i]
        sum1_y += Y1[i]
        sum1_xsq += X1[i] * X1[i]
        sum1_xy += X1[i] * Y1[i]
    # Summation for second set of points
    for j in range(num2):
        sum2_x += X2[j]
        sum2_y += Y2[j]
        sum2_xsq += X2[j] * X2[j]
        sum2_xy += X2[j] * Y2[j]
    # Calculation of coefficients
    c11 = sum1_xsq + num2 * xc * xc
    c12 = xc * sum2_x - num2 * xc * xc
    c13 = sum1_x + xc * num2
    c14 = -sum1_xy - sum2_y * xc
    c21 = xc * sum2_x - num2 * xc * xc
    c22 = sum2_xsq - 2 * xc * sum2_x + num2 * xc * xc
    c23 = sum2_x - num2 * xc
    c24 = -sum2_xy + xc * sum2_y
    c31 = sum1_x + num2 * xc
    c32 = sum2_x - num2 * xc
    c33 = n
    c34 = -sum1_y - sum2_y
    # Determinant calculation
    H = c11 * c22 * c33 + c12 * c23 * c31 + c21 * c32 * c13 - c13 * c22 * c31 - c12 * c21 * c33 - c11 * c23 * c32
    # Checking determinant and calculating slopes and intercepts
    if H == 0: # not possible with real data.
        a1 = b1 = a2 = b2 = -1
        err = 12345678
    else:
        a1 = -(c14 * c22 * c33 + c12 * c23 * c34 + c24 * c32 * c13 - c13 * c22 * c34 - c12 * c24 * c33 - c23 * c32 * c14) / H
        a2 = -(c11 * c24 * c33 + c21 * c34 * c13 + c14 * c23 * c31 - c13 * c31 * c24 - c11 * c23 * c34 - c14 * c21 * c33) / H
        b1 = -(c11 * c22 * c34 + c21 * c32 * c14 + c12 * c24 * c31 - c14 * c22 * c31 - c12 * c21 * c34 - c11 * c32 * c24) / H
        b2 = xc * (a1 - a2) + b1
        for i in range(num1):
            err += (a1 * X1[i] + b1 - Y1[i])**2
        for j in range(num2):
            err += (a2 * X2[j] + b2 - Y2[j])**2
    return a1, b1, a2, b2, err

def getAvailableFreqs():
    """
    Get all available frequency values.
    Run "nvidia-smi -q -d SUPPORTED_CLOCKS" to get available frequencies and adjust this function.
    """
    available_freqs = []
    if MACHINE == "v100-maxq":
        """
        freq = 135
        seven = True
        available_freqs.append(freq)
        while freq <= 1440:
            if seven:
                freq += 7
                seven = False
            else:
                freq += 8
                seven = True
            if freq <= 1440:
                available_freqs.append(freq)
        """
        available_freqs = [135, 142, 150, 157, 165, 172, 180, 187, 195, 202, 210, 217, 225, 232, 240, 247, 255, 262, 270, 277, 285, 292, 300, 307, 315, 322, 330, 337, 345, 352, 360, 367, 375, 382, 390, 397, 405, 412, 420, 427, 435, 442, 450, 457, 465, 472, 480, 487, 495, 502, 510, 517, 525, 532, 540, 547, 555, 562, 570, 577, 585, 592, 600, 607, 615, 622, 630, 637, 645, 652, 660, 667, 675, 682, 690, 697, 705, 712, 720, 727, 735, 742, 750, 757, 765, 772, 780, 787, 795, 802, 810, 817, 825, 832, 840, 847, 855, 862, 870, 877, 885, 892, 900, 907, 915, 922, 930, 937, 945, 952, 960, 967, 975, 982, 990, 997, 1005, 1012, 1020, 1027, 1035, 1042, 1050, 1057, 1065, 1072, 1080, 1087, 1095, 1102, 1110, 1117, 1125, 1132, 1140, 1147, 1155, 1162, 1170, 1177, 1185, 1192, 1200, 1207, 1215, 1222, 1230, 1237, 1245, 1252, 1260, 1267, 1275, 1282, 1290, 1297, 1305, 1312, 1320, 1327, 1335, 1342, 1350, 1357, 1365, 1372, 1380, 1387, 1395, 1402, 1410, 1417, 1425, 1432, 1440]
    elif MACHINE == "v100-300w":
        """
        freq = 135
        seven = True
        available_freqs.append(freq)
        while freq <= 1530:
            if seven:
                freq += 7
                seven = False
            else:
                freq += 8
                seven = True
            if freq <= 1530:
                available_freqs.append(freq)
        """
        available_freqs = [135, 142, 150, 157, 165, 172, 180, 187, 195, 202, 210, 217, 225, 232, 240, 247, 255, 262, 270, 277, 285, 292, 300, 307, 315, 322, 330, 337, 345, 352, 360, 367, 375, 382, 390, 397, 405, 412, 420, 427, 435, 442, 450, 457, 465, 472, 480, 487, 495, 502, 510, 517, 525, 532, 540, 547, 555, 562, 570, 577, 585, 592, 600, 607, 615, 622, 630, 637, 645, 652, 660, 667, 675, 682, 690, 697, 705, 712, 720, 727, 735, 742, 750, 757, 765, 772, 780, 787, 795, 802, 810, 817, 825, 832, 840, 847, 855, 862, 870, 877, 885, 892, 900, 907, 915, 922, 930, 937, 945, 952, 960, 967, 975, 982, 990, 997, 1005, 1012, 1020, 1027, 1035, 1042, 1050, 1057, 1065, 1072, 1080, 1087, 1095, 1102, 1110, 1117, 1125, 1132, 1140, 1147, 1155, 1162, 1170, 1177, 1185, 1192, 1200, 1207, 1215, 1222, 1230, 1237, 1245, 1252, 1260, 1267, 1275, 1282, 1290, 1297, 1305, 1312, 1320, 1327, 1335, 1342, 1350, 1357, 1365, 1372, 1380, 1387, 1395, 1402, 1410, 1417, 1425, 1432, 1440, 1447, 1455, 1462, 1470, 1477, 1485, 1492, 1500, 1507, 1515, 1522, 1530]
    elif MACHINE == "a100-insp":
        """
        freq = 210
        available_freqs.append(freq)
        while freq <= 1410:
            freq += 15
            if freq <= 1410:
                available_freqs.append(freq)
        """
        available_freqs = [210, 225, 240, 255, 270, 285, 300, 315, 330, 345, 360, 375, 390, 405, 420, 435, 450, 465, 480, 495, 510, 525, 540, 555, 570, 585, 600, 615, 630, 645, 660, 675, 690, 705, 720, 735, 750, 765, 780, 795, 810, 825, 840, 855, 870, 885, 900, 915, 930, 945, 960, 975, 990, 1005, 1020, 1035, 1050, 1065, 1080, 1095, 1110, 1125, 1140, 1155, 1170, 1185, 1200, 1215, 1230, 1245, 1260, 1275, 1290, 1305, 1320, 1335, 1350, 1365, 1380, 1395, 1410]
    else:
        print("Should edit function getAvailableFreqs() to add available frequencies.")
        sys.exit(0)
    numAvailableFreqs = len(available_freqs)
    return available_freqs, numAvailableFreqs

keep_running = True
def signal_handler(signum, frame):
    global keep_running
    keep_running = False

def main():
    print("MACHINE=%s" % MACHINE)
    print("selectGPU=%d" % selectGPU)
    print("policy=%s" % policy)
    print("loopDelay=%f" % loopDelay)
    global probDelay
    print("probDelay=%f" % probDelay)
    print("update_frequency=%d" % update_frequency)
    print("useFreqCap=%d" % useFreqCap)
    if verbose:
        print("Warning: verbose set as true.")
    if selectGPU != -1:
        print("Warning: only GPU %d is selected." % selectGPU)
    if skipSetFreq:
        print("Warning: skipSetFreq set as true.")
    # Instantiate a DcgmReader object
    print("starting DcgmReader..")
    keep_time = 3600.0 # Max time in dcgm to keep data from NVML, in seconds. Default is 3600.0 (1 hour)
    dr = DcgmReader.DcgmReader(fieldIds=customFields, updateFrequency=update_frequency, maxKeepAge=keep_time, ignoreList=[], gpuIds=None, fieldGroupName="dcgm_fieldgroupdata")

    print("starting nvml..")
    pynvml.nvmlInit()
    print("Driver Version:", pynvml.nvmlSystemGetDriverVersion())
    deviceCount = pynvml.nvmlDeviceGetCount()
    availableFreqs, numAvailableFreqs = getAvailableFreqs()

    gpuUtils, moving_gmemUtils, moving_gmemUtils_sq, gmemUtils, gPowers, list_gpuUtils, list_gmemUtils, list_gPowers = [], [], [], [], [], [], [], []
    for i in range(deviceCount):
        moving_gmemUtils.append(MovingAverage(movingAvg_windowSize))
        moving_gmemUtils_sq.append(MovingAverage(movingAvg_windowSize))
        gpuUtils.append([])
        gmemUtils.append([])
        gPowers.append([])
    for i in range(numProbFreq):
        list_gpuUtils.append([])
        list_gmemUtils.append([])
        list_gPowers.append([])
    setFreq = [-1]*deviceCount
    optimizedFreqs = [maxFreq]*deviceCount
    gmemutil_moving_avg = [-1]*deviceCount
    avg_gpuUtils = [-1]*numProbFreq # record the average gpuUtil for each probing frequency.
    avg_gmemUtils = [-1]*numProbFreq # record the average gmemUtil for each probing frequency.
    avg_gPowers = [-1]*numProbFreq # record the average gPower for each probing frequency.
    modelPerf = [-1]*numProbFreq # record model-estimated performance.
    powerEffici = [-1]*numProbFreq # record power efficiency.
    allModelPerf = [-1]*numAvailableFreqs
    allModelPower = [-1]*numAvailableFreqs
    allPowerEffici = [-1]*numAvailableFreqs

    cycle = 0
    printCol = 1
    initialLoop = True
    probPhase = numProbFreq * numProbRep # This value will decrease in the loop.
    accumuTime = 0
    signal.signal(signal.SIGINT, signal_handler)
    while keep_running:
        try:
            starttime = datetime.now()
            if policy in ["UtilizScale"]:
                cycle += 1
                if cycle == probInterval:
                    cycle = 0
            
            # Buggy probabilistically. Do not use.
            #infoCount, infos = pynvml.nvmlDeviceGetComputeRunningProcesses(device)
            #print("Running process count: %d" % infoCount)

            # Read DCGM data for all GPUs.
            data = dr.GetLatestGpuValuesAsFieldNameDict()
            thisTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            # Calculate freq.
            if policy == "MaxFreq":
                setFreq = [maxFreq]*deviceCount
                applyFreqSet = initialLoop # only applyFreqSet in 1st loop.
            elif policy == "EfficientFix":
                setFreq = [freqAvgEff]*deviceCount
                applyFreqSet = initialLoop # only applyFreqSet in 1st loop.
            elif policy == "NVboost":
                setFreq = [-1]*deviceCount
                applyFreqSet = False # leave control to the default freq controller.
            elif policy == "UtilizScale":
                if cycle == 1:
                    # Prob the utilization at max frequency.
                    setFreq = [maxFreq]*deviceCount
                    applyFreqSet = True
                elif cycle == 2:
                    for i in range(deviceCount):
                        grAct = data[i]["gr_engine_active"]
                        # Set freq proportional to gpu util, but bounded by minSetFreq from below.
                        optimizedFreqs[i] = max(minSetFreq, grAct*maxFreq)
                        # Loop through all available frequency values to find the nearest larger one.
                        for iAvail in range(numAvailableFreqs - 1, -1, -1):
                            if availableFreqs[iAvail] < optimizedFreqs[i]:
                                if iAvail < numAvailableFreqs - 1:
                                    optimizedFreqs[i] = availableFreqs[iAvail + 1]
                                else:
                                    optimizedFreqs[i] = availableFreqs[iAvail]
                                break
                        setFreq[i] = optimizedFreqs[i]
                    applyFreqSet = True
                else:
                    setFreq = [-1]*deviceCount
                    applyFreqSet = False
            elif policy == "Assure":
                # GEEPAFS policy.
                for i in range(deviceCount):
                    # Note that all DCGM activity values are <=1.
                    grAct = data[i]["gr_engine_active"]
                    memValue = data[i]["dram_active"]
                    powerValue = data[i]["power_usage"]
                    moving_gmemUtils[i].addData(memValue)
                    moving_gmemUtils_sq[i].addData(memValue*memValue)
                    gmemutil_moving_avg[i] = moving_gmemUtils[i].getMovingAverage()
                    if probPhase >= 0 and probPhase < numProbRec:
                        # During probing phase, record gpu memory bandwidth utilization
                        # Be careful that the recorded util values corresponds to the last frequency setting in the previous loop.
                        gpuUtils[i].append(grAct)
                        gmemUtils[i].append(memValue)
                        gPowers[i].append(powerValue)
                    if probPhase > 0:
                        # in probing phase, change gpu freqs to prob the response of gpu mem bw utils.
                        iprob = numProbRec - probPhase # iprob start at 0 and increase.
                        reminder = iprob % (2*numProbFreq)
                        if reminder < numProbFreq:
                            setFreq[i] = probFreqs[reminder]
                        else:
                            setFreq[i] = probFreqs[2*numProbFreq-1-reminder]
                    elif probPhase == 0:
                        # keep the last freq setting because the optimized freq has not been calculated.
                        iprob = numProbRec - 1 # cannot be combined with the probPhase > 0 case.
                        reminder = iprob % (2*numProbFreq)
                        if reminder < numProbFreq:
                            setFreq[i] = probFreqs[reminder]
                        else:
                            setFreq[i] = probFreqs[2*numProbFreq-1-reminder]
                    else:
                        if skipSetFreq:
                            setFreq[i] = maxFreq # only for measuring policy cost.
                        else:
                            setFreq[i] = optimizedFreqs[i] # calculated when probPhase==0.
                    if probPhase >= -1:
                        applyFreqSet = True
                    else:
                        applyFreqSet = False # not apply freq set to reduce delay.

            # Set GPU freq using NVML. Here, setFreq is a list with freq targets for all GPUs.
            # Note: Avoiding unnecessary freqset can significantly reduce delay.
            # Note: When power is high, actual freq may be consistently lower than setFreq due to thermal throttling.
            if applyFreqSet:
                for i in range(deviceCount):
                    if selectGPU == -1 or i == selectGPU:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        if setAppFreq:
                            pynvml.nvmlDeviceSetApplicationsClocks(handle, setMemFreq, setFreq[i])
                        else:
                            pynvml.nvmlDeviceSetGpuLockedClocks(handle, setFreq[i], setFreq[i])
                                # both the lower and higher limit are set here.

            # Print table column names at the beginning.
            if printCol:
                headline = "time"
                for fieldName in data[0]:
                    headline += ",%s" % (fieldName)
                headline += ",setFreq"
                print(headline)
                printCol = 0

            # Print the data.
            line = thisTime
            for gpuId in data:
                if selectGPU == -1 or gpuId == selectGPU:
                    for fieldName in data[gpuId]:
                        line += ",%s" % (data[gpuId][fieldName])
                    line += ",%d" % (setFreq[gpuId])
            print(line)

            # still inside the while loop..

            # In Assure, if just finished probing phase, fit the performance model and calculate the optimized freq.
            if policy == "Assure" and probPhase == 0:
                for i in range(deviceCount):
                    # print the gemUtils array.
                    if verbose:
                        if selectGPU == -1 or i == selectGPU:
                            datastr = " ".join(["%f" % x for x in gmemUtils[i]])
                            print("Device %d mem bw util: %s" % (i, datastr))
                    # Calculate avg_gmemUtils and avg_gPowers.
                    x, y = [],[] # x, y inputs for the single linear model.
                    for j in range(numProbFreq):
                        list_gpuUtils[j] = []
                        list_gmemUtils[j] = []
                        list_gPowers[j] = []
                    recordLen = len(gmemUtils[i])
                    for j in range(recordLen):
                        reminder = j % (2*numProbFreq)
                        y.append(gmemUtils[i][j])
                        if reminder >= numProbFreq:
                            reminder = 2*numProbFreq-1-reminder
                        x.append(probFreqs[reminder])
                        list_gpuUtils[reminder].append(gpuUtils[i][j])
                        list_gmemUtils[reminder].append(gmemUtils[i][j])
                        list_gPowers[reminder].append(gPowers[i][j])
                    # calculate avg and standard error.
                    avg_gpuUtils = [ np.mean(list_gpuUtils[j]) for j in range(numProbFreq)]
                    avg_gmemUtils = [ np.mean(list_gmemUtils[j]) for j in range(numProbFreq)]
                    avg_gPowers = [ np.mean(list_gPowers[j]) for j in range(numProbFreq)]
                    # print data.
                    if verbose:
                        if selectGPU == -1 or i == selectGPU:
                            datastr = " ".join(["%f" % x for x in avg_gpuUtils])
                            print("Device %d: avg gpu util at each freq: %s" % (i, datastr))
                            datastr = " ".join(["%f" % x for x in avg_gmemUtils])
                            print("Device %d: avg mem util at each freq: %s" % (i, datastr))
                            datastr = " ".join(["%f" % x for x in avg_gPowers])
                            print("Device %d: avg gpu power at each freq: %s" % (i, datastr))
                            datastr = " ".join(["%f" % (100*avg_gmemUtils[ix]/avg_gPowers[ix]) for ix in range(numProbFreq)])
                            print("Device %d: effici(=100*gmemutil/gpower): %s" % (i, datastr))
                    # Fit the model with 1-fold foldline regression.
                    if useRegression:
                        # fit model when all the gmem util is nonzero.
                        if sum(y)/len(y) >= gmembwLowThres:
                            # Build the performance and power efficiency model to optimize frequency.
                            # Case 1: fit the points with a single linear model.
                            slope_Opt, intercept_Opt, regErr = linearRegression(x, y)
                            regErrMin = regErr
                            turn_Opt = 0
                            if verbose:
                                print("Device %d: turn=non, slope=%f, intercept=%f, regErr=%f" % (i, slope_Opt, intercept_Opt, regErr))
                            # Case 2: Partition the points and fit the points with two linear models connected by a turning point.
                            for turn in range(2, numProbFreq-1):
                                # "turn" marks how many points are to fit the 1st linear model.
                                # Partition the points to fit two linear models.
                                # *1 for lower frequency, and *2 for higher frequency.
                                # We set 2 <= turn <= numProbFreq-2, otherwise we encounter fitting a model with only one point.
                                x1, y1, x2, y2 = [],[],[],[]
                                for j in range(numProbRec):
                                    reminder = j % (2*numProbFreq)
                                    if reminder < numProbFreq:
                                        ifreq = reminder
                                    else:
                                        ifreq = 2*numProbFreq-1-reminder
                                    if ifreq < turn:
                                        x1.append(probFreqs[ifreq])
                                        y1.append(gmemUtils[i][j])
                                    else:
                                        x2.append(probFreqs[ifreq])
                                        y2.append(gmemUtils[i][j])
                                slope1, intercept1, regErr1 = linearRegression(x1, y1)
                                slope2, intercept2, regErr2 = linearRegression(x2, y2)
                                if slope2 != slope1:
                                    freq_cross = (intercept1-intercept2) / (slope2-slope1)
                                else:
                                    freq_cross = -1
                                if freq_cross >= probFreqs[turn-1] and freq_cross <= probFreqs[turn]:
                                    # this fitted fold-line is valid.
                                    regErr = regErr1 + regErr2
                                else:
                                    # re-fit the fold-line and let the cross to happen at probFreqs[turn-1].
                                    freq_cross = probFreqs[turn-1]
                                    slope1, intercept1, slope2, intercept2, regErr = foldlineRegression(freq_cross, x1, y1, x2, y2)
                                if verbose:
                                    print("Device %d: turn=%d, slope1=%f, intercept1=%f, slope2=%f, intercept2=%f, regErr=%f" % (i, turn, slope1, intercept1, slope2, intercept2, regErr))
                                if slope1 <= slope2:
                                    if verbose:
                                        print("slope1 <= slope2, abandon this partition.")
                                else:
                                    # record if it is optimal.
                                    if regErr < regErrMin:
                                        turn_Opt = turn
                                        regErrMin = regErr
                                        slope1_Opt, intercept1_Opt = slope1, intercept1
                                        slope2_Opt, intercept2_Opt = slope2, intercept2
                                        freq_cross_Opt = freq_cross
                                        if verbose:
                                            print("Better model found.")
                                    else:
                                        if verbose:
                                            print("Larger reg err, not used.")
                            # if regression error too large, do not use regression model. Set freq by util.
                            if regErrMin > numProbRec * regErrThres:
                                if verbose:
                                    print("All regression err are larger than %f, discard models." % (numProbRec * regErrThres))
                                skipmodel = True
                                # set a high frequency for assurance.
                                freqBound = maxFreq # will be bounded by freqCap later.
                                freqEff = freqAvgEff
                            else:
                                skipmodel = False
        
                            if not skipmodel:
                                # Calculate power efficiency based on modeled performance and power.
                                if calcAllEffici:
                                    # Estimate power efficiency at all available frequencies and >= minSetFreq.
                                    if turn_Opt == 0:
                                        for j in range(numAvailableFreqs):
                                            if availableFreqs[j] >= minSetFreq:
                                                if slope_Opt > 0:  # assume performance correlates to gmemutil.
                                                    allModelPerf[j] = slope_Opt * availableFreqs[j] + intercept_Opt
                                                else:  # assume the lowest probing frequency's performance maintains.
                                                    allModelPerf[j] = slope_Opt * probFreqs[0] + intercept_Opt
                                    else:  # fold-line model.
                                        # slope1_Opt for lower frequency model, and slope2_Opt for higher.
                                        if slope1_Opt > 0 and slope2_Opt > 0:
                                            for j in range(numAvailableFreqs):
                                                if availableFreqs[j] >= minSetFreq:
                                                    if availableFreqs[j] >= freq_cross_Opt:
                                                        # model for higher frequency.
                                                        allModelPerf[j] = slope2_Opt * availableFreqs[j] + intercept2_Opt
                                                    else:
                                                        # model for lower frequency.
                                                        allModelPerf[j] = slope1_Opt * availableFreqs[j] + intercept1_Opt
                                        elif slope2_Opt <= 0 and slope1_Opt > 0:  # maximum is in the middle.
                                            for j in range(numAvailableFreqs):
                                                if availableFreqs[j] >= minSetFreq:
                                                    if availableFreqs[j] < freq_cross_Opt:
                                                        # model for lower frequency.
                                                        allModelPerf[j] = slope1_Opt * availableFreqs[j] + intercept1_Opt
                                                    else:
                                                        # use the estimated performance at cross.
                                                        freq_cross = (intercept1_Opt - intercept2_Opt) / (slope2_Opt - slope1_Opt)
                                                        allModelPerf[j] = (slope2_Opt * intercept1_Opt - slope1_Opt * intercept2_Opt) / (slope2_Opt - slope1_Opt)
                                        else:  # slope1_Opt <= 0.
                                            for j in range(numAvailableFreqs):
                                                if availableFreqs[j] >= minSetFreq:
                                                    # estimate performance using the lowest frequency.
                                                    allModelPerf[j] = slope1_Opt * probFreqs[0] + intercept1_Opt
                            
                                    #end if model with turing point.
                            
                                    # fit freq-power model with regression.
                                    x3, y3 = [],[]
                                    for j in range(numProbRec):
                                        y3.append(gPowers[i][j])
                                        reminder = j % (2 * numProbFreq)
                                        if reminder < numProbFreq:
                                            x3.append(probFreqs[reminder])
                                        else:
                                            x3.append(probFreqs[2 * numProbFreq - 1 - reminder])
                                    coefs = np.polyfit(x3, y3, 3)
                                    if verbose:
                                        print("Power model coefs: " + str(coefs))
                                    for j in range(numAvailableFreqs):
                                        if availableFreqs[j] >= minSetFreq:
                                            f = availableFreqs[j]
                                            allModelPower[j] = coefs[0] * f**3 + coefs[1] * f**2 + coefs[2] * f + coefs[3]
                            
                                    # calculate the power efficiency.
                                    for j in range(numAvailableFreqs):
                                        if availableFreqs[j] >= minSetFreq:
                                            allPowerEffici[j] = 100 * allModelPerf[j] / allModelPower[j]
                                    if verbose:
                                        print("Device %d: modeled performance:" % i, end="")
                                        for j in range(numAvailableFreqs):
                                            if availableFreqs[j] >= minSetFreq:
                                                print(" %.6f" % (allModelPerf[j]), end="")  # print from low to high frequency.
                                        print("\nDevice %d: modeled power:" % i, end="")
                                        for j in range(numAvailableFreqs):
                                            if availableFreqs[j] >= minSetFreq:
                                                print(" %.6f" % (allModelPower[j]), end="")  # print from low to high frequency.
                                        print("\nDevice %d: power efficiency:" % i, end="")
                                        for j in range(numAvailableFreqs):
                                            if availableFreqs[j] >= minSetFreq:
                                                print(" %.6f" % (allPowerEffici[j]), end="")  # print from low to high frequency.
                                        print()
                            
                                    # find the most power efficient frequency.
                                    for j in range(numAvailableFreqs):
                                        if availableFreqs[j] == minSetFreq:
                                            mostEffici = allPowerEffici[j]
                                            mostEfficiFreq = availableFreqs[j]
                                        elif availableFreqs[j] > minSetFreq:
                                            if allPowerEffici[j] > mostEffici:
                                                mostEffici = allPowerEffici[j]
                                                mostEfficiFreq = availableFreqs[j]
                                    freqEff = mostEfficiFreq
                                    if verbose:
                                        print("Device %d: max efficiency %.6f at frequency %d MHz." % (i, mostEffici, mostEfficiFreq))
                                else:
                                    # Estimate power efficiency only at the probed frequencies.
                                    if turn_Opt == 0:
                                        for j in range(numProbFreq):
                                            if slope_Opt > 0:  # assume performance correlates to gmemutil.
                                                modelPerf[j] = slope_Opt * probFreqs[j] + intercept_Opt
                                            else:  # assume the lowest frequency's performance is maximal.
                                                modelPerf[j] = slope_Opt * probFreqs[0] + intercept_Opt
                                    else:  # fold-line model.
                                        # slope1_Opt for lower frequency, and slope2_Opt for higher.
                                        if slope1_Opt > 0 and slope2_Opt > 0:
                                            for j in range(numProbFreq):
                                                if j >= turn_Opt:
                                                    # model for higher frequency.
                                                    modelPerf[j] = slope2_Opt * probFreqs[j] + intercept2_Opt
                                                else:
                                                    # model for lower frequency.
                                                    modelPerf[j] = slope1_Opt * probFreqs[j] + intercept1_Opt
                                        elif slope2_Opt <= 0 and slope1_Opt > 0:  # maximum is in the middle.
                                            for j in range(numProbFreq):
                                                if j < turn_Opt:
                                                    # model for lower frequency.
                                                    modelPerf[j] = slope1_Opt * probFreqs[j] + intercept1_Opt
                                                else:
                                                    # use the estimated performance at cross.
                                                    freq_cross = (intercept1_Opt - intercept2_Opt) / (slope2_Opt - slope1_Opt)
                                                    modelPerf[j] = (slope2_Opt * intercept1_Opt - slope1_Opt * intercept2_Opt) / (slope2_Opt - slope1_Opt)
                                        else:  # slope1_Opt <= 0.
                                            for j in range(numProbFreq):
                                                # estimate performance using the lowest frequency.
                                                modelPerf[j] = slope1_Opt * probFreqs[0] + intercept1_Opt
                                    
                                    # calculate the power efficiency.
                                    for j in range(numProbFreq):
                                        powerEffici[j] = 100 * modelPerf[j] / avg_gPowers[j]
                                    
                                    if verbose:
                                        print("Device %d: modeled performance:" % i, end="")
                                        for j in range(numProbFreq):
                                            print(" %.6f" % (modelPerf[j]), end="")  # print from low to high frequency.
                                        print()
                                        print("Device %d: power efficiency:" % i, end="")
                                        for j in range(numProbFreq):
                                            print(" %.6f" % (powerEffici[j]), end="")  # print from low to high frequency.
                                        print()
                                    
                                    # find the most power efficient frequency.
                                    mostEffici = powerEffici[0]
                                    mostEfficiFreq = probFreqs[0]
                                    for j in range(1, numProbFreq):
                                        if powerEffici[j] > mostEffici:
                                            mostEffici = powerEffici[j]
                                            mostEfficiFreq = probFreqs[j]
                                    
                                    freqEff = mostEfficiFreq
                                    if verbose:
                                        print("Device %d: max efficiency %.6f at frequency %d MHz." % (i, mostEffici, mostEfficiFreq))
                            
                                # calculate critical frequency bounded by performance constraint using gmem util model.
                                if turn_Opt == 0: # if a single linear model is optimal.
                                    if slope_Opt > 0:
                                        freq_perfBound = (perfThres * (slope_Opt * maxFreq + intercept_Opt) - intercept_Opt) / slope_Opt
                                    else: # lower frequency is better.
                                        freq_perfBound = probFreqs[0]
                                    if verbose:
                                        print("Performance estimated by single linear model.")
                                else: # if fold-line model.
                                    if slope1_Opt > 0:
                                        if slope2_Opt > 0:
                                            criticalPerf = perfThres * (slope2_Opt * maxFreq + intercept2_Opt)
                                            freq_perfBound = (criticalPerf - intercept2_Opt) / slope2_Opt
                                            freq_cross = (intercept1_Opt - intercept2_Opt) / (slope2_Opt - slope1_Opt)
                                            if freq_perfBound <= freq_cross:
                                                # should use low-freq-model instead.
                                                freq_perfBound = (criticalPerf - intercept1_Opt) / slope1_Opt
                                                if verbose:
                                                    print("Performance assurance satisfied at low-segment.")
                                            else:
                                                if verbose:
                                                    print("Performance assurance satisfied at high-segment.")
                                        elif slope2_Opt <= 0:
                                            freq_cross = (intercept1_Opt - intercept2_Opt) / (slope2_Opt - slope1_Opt)
                                            criticalPerf = perfThres * (slope1_Opt * freq_cross + intercept1_Opt)
                                            if verbose:
                                                print("Performance saturation predicted at %.1f MHz." % freq_cross)
                                            freq_perfBound = (criticalPerf - intercept1_Opt) / slope1_Opt
                                            if verbose:
                                                print("Performance assurance satisfied at low-segment.")
                                    # end if slope1_Opt > 0.
                                    else:
                                        if verbose:
                                            print("Performance saturation predicted at %d MHz." % (probFreqs[0]))
                                        freq_perfBound = probFreqs[0]
                                # end if fold-line model.
                                
                                if verbose:
                                    print("Device %d: performance assurance achieved at %.1f MHz." % (i, freq_perfBound))
                                
                                freqBound = freq_perfBound
                        else:
                            if verbose:
                                print("Device %d: avg mem bw <%.2f, will set frequency by util." % (i, gmembwLowThres))
                            freqBound = maxFreq # will be bounded by freqCap later.
                            freqEff = freqAvgEff # on A100, gmemutil may be always 0 for a few apps.
                    # end if useRegression.
                    else:
                        # set freqBound as the freq with max gmemUtil.
                        max_gmem = avg_gmemUtils[0]
                        max_gmem_freq = probFreqs[0]
                        for j in range(1, numProbFreq):
                            if avg_gmemUtils[j] > max_gmem:
                                max_gmem = avg_gmemUtils[j]
                        for j in range(numProbFreq):
                            if avg_gmemUtils[j] >= max_gmem * 0.99:
                                max_gmem_freq = probFreqs[j]
                                break
                        freqBound = max_gmem_freq
                        freqEff = freqAvgEff
                    
                    if useFreqCap:
                        freqCap = perfThres*maxFreq*max(avg_gpuUtils)
                        freqPerf = min(freqBound, freqCap)
                        if verbose and freqBound > freqCap:
                            if selectGPU == -1 or i == selectGPU:
                                print("Device %d: set frequency %.1f capped by gpu util." % (i, freqCap))
                    else:
                        freqPerf = freqBound
                    
                    freqOpt = max(freqPerf, freqEff)
                    
                    # set optimized freq by looking through the available freq list.
                    freqOpt = max(freqOpt, float(minSetFreq))
                    freqOpt = min(freqOpt, float(maxFreq))
                    if verbose:
                        if selectGPU == -1 or i == selectGPU:
                            print("Device %d, freqBound=%d, freqEff=%d, freqOpt=%d." % (i, freqBound, freqEff, freqOpt))
                    for iAvail in range(numAvailableFreqs - 1, -1, -1):
                        if availableFreqs[iAvail] < freqOpt:
                            if iAvail < numAvailableFreqs - 1:
                                optimizedFreqs[i] = availableFreqs[iAvail + 1]
                            else:
                                optimizedFreqs[i] = availableFreqs[iAvail]
                            break
                # end for device index.
                if verbose:
                    print("Optimized frequencies:", end="")
                    for i in range(deviceCount):
                        print("\t%d" % (optimizedFreqs[i]), end="")
                    print()
            # end if just finished probing.
            endtime = datetime.now()
            duration = (endtime - starttime).total_seconds()
            if verbose:
                print("latency: %.3f seconds" % duration)
            if duration < loopDelay:
                time.sleep(loopDelay - duration)
                addTime = loopDelay
            else:
                addTime = duration
            if policy == "Assure":
                # determine whether or not enter the probing phase.
                if accumuTime >= probDelay:
                    # Every probDelay seconds, start probing.
                    if sum(gmemutil_moving_avg) >= gmembwLowThres:
                        if verbose:
                            print("Probing phase start at %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        probPhase = numProbRec  # probPhase marks the execution of the probing phase.
                    else:
                        probPhase = -2
                        if verbose:
                            print("Negligible avg gmemutil, <%.2f. Probing omitted." % gmembwLowThres)
                    for i in range(deviceCount):
                        gpuUtils[i], gmemUtils[i], gPowers[i] = [], [], [] # reset the lists.
                    accumuTime = 0  # reset accumuTime.
                else:
                    if probPhase > -1:
                        accumuTime = 0  # only accumulate time after probing phase.
                    else:
                        accumuTime += addTime
                    if probPhase > -99:  # use a lower limit -99 to prevent overflow.
                        probPhase -= 1
            # end if Assure.
            initialLoop = False

        except Exception as e:
            print("An exception occurred: %s" % e)

    #except KeyboardInterrupt:
    print("Ctrl-C is pressed. Stopping.")
    print("Resetting GPU frequency..")
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        pynvml.nvmlDeviceResetGpuLockedClocks(handle)
        pynvml.nvmlDeviceResetApplicationsClocks(handle)
    print("NVML shutting down..")
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()

