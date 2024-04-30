# GEEPAFS

This repository contains the source code of the GEEPAFS policy, as well as examplar scripts to launch experiments and post-process results.

## Requirements

This code targets NVIDIA GPU V100 or A100 on Linux platforms. NVIDIA CUDA should be fully installed on the system. Root privileges are necessary in order to apply GPU frequency setting. For the python version `dvfsPython.py`, DCGM should also be installed.

Note that some V100/A100 GPUs' max frequency may be slightly different than the values defined in our codes. In that case, corresponding variables should be adjusted.

GEEPAFS can also work for GPU types other than V100/A100 after small edits on variables. To work for another NVIDIA GPU, the constants in `dvfs.c` / `dvfsPython.py` including minSetFreq, freqAvgEff, maxFreq, setMemFreq, numAvailableFreqs, numProbFreq, probFreqs, and the function getAvailableFreqs() need to be updated. To work for AMD/Intel/... GPUs, the NVML API calls for metric reading and frequency tuning should also be replaced by corresponding API calls.

## C-NVML version vs Python-DCGM version

We provide a new python version of our policy in file `dvfsPython.py`. Compared to the original C version `dvfs.c` which uses NVML to collect hardware metrics, the python version uses NVIDIA DCGM to collect metrics. (Both versions still use NVML to tune frequency.)

As DCGM shows much better accuracy and lower latency than NVML in metric collection, this python version is expected to perform better.

## Setup and Run

To use the C version `dvfs.c`:
- To run the GEEPAFS policy, first open `dvfs.c` and select the correct GPU type by editing the `#define` lines at the front.
- Then, compile `dvfs.c` by executing `make`. Note that `CUDA_PATH` in the Makefile may need to be changed if cuda cannot be found in its default place.
- After compilation, run GEEPAFS with default settings by the command `sudo ./dvfs mod Assure p90`. This command runs the GEEPAFS policy with a performance constraint of 90%. Note that root privileges are necessary in applying frequency tuning. This program runs endlessly by default. Press ctrl-c to stop.
- To run a baseline policy, use the command `sudo ./dvfs mod MaxFreq`, where the name `MaxFreq` can also be replaced by `NVboost`, `EfficientFix`, or `UtilizScale`.

To use the python version `dvfsPython.py`:
- Select the correct GPU type by editing the `MACHINE =` line.
- Run with default settings by the command `sudo python dvfsPython.py Assure 90`. The last number 90 represents the performance constraint 90%. Root privileges are necessary in applying frequency tuning.
- Require install DCGM and pynvml. By default, the DcgmReader class of DCGM is installed in `/usr/local/dcgm/bindings/python3/`. If installed to another place, that path should be added using `sys.path.append()` function like our example.
- Python 3 is preferred, and Python 2 may not work. Note: executing with `sudo python` may lead to a different version than `python`. You can use the full path `sudo /path/to/your/python` to avoid that issue.

## Launch Experiments

Scripts to launch experiments are provided in `runExp.py`.
- To use the script, first go to folder `cuda_samples/benchmarks/` and use `make` in subfolders to compile each applications one by one. The files in the `cuda_samples` folder are from NVIDIA CUDA Code Samples and are copied here only to facilitate the running of testing experiments. We have made minor changes to the application source codes to extend their execution time.
- After benchmark compilation, launch experiments by the command `sudo python3 runExp.py`. It will automatically launch the GEEPAFS daemon (C version) and the benchmarks as subprocesses. Results are saved into the `./output/` folder.
- To launch experiments with the GEEPAFS python version, please manually replace the `./dvfs` launch code in the script `runExp.py`.

Other benchmarks may also be added into experiments in similar ways. This package does not include more benchmarks as they usually require more steps in compilation and larger datasets (e.g., ImageNet2012 dataset occupies 150 GB).

## Post-Processing

Scripts to post-process the data generated by `runExp.py` are provided in `postprocessing.py`.
By default, the script reads the examplar files `allApps_Assure_p90_2iter_demo.out`, `dvfs_Assure_p90_demo.out`, and output processed files in .csv format. The script calculates the average performance, average power usage, average energy efficiency, etc. for each application.
Edit the script to process other files or other benchmarks.

If experimenting with GEEPAFS python version, the post-processing scripts need to be slightly modified to match the `dvfsPython.py` output format.

## Latency Measurement

In the `./latency/` folder, we provide a small program to show how to measure the latency of NVML's metric reading and frequency tuning. More instructions can be found in `./latency/measure_latency.c`.

## Debugging

- If executing `dvfs.c` and encountering error "Failed to set frequency for GPU 0: Invalid Argument", it means the frequency value (either the GPU frequency or the GPU memory frequency) to be set is not supported. Please execute command `nvidia-smi -q -d SUPPORTED_CLOCKS` to check the supported frequency values, and adjust the hard-coded frequency values in `dvfs.c`.
- We notice that workload cudaTensorCoreGemm's performance and power are significantly reduced when compiled using CUDA 12. This issue isn't seen in CUDA 11. We are unsure about the root cause.

## On-going Work

We are working on improving the accuracy and flexibility of our performance modeling component. An updated version of our GPU DVFS policy will be released in the future.

## Reference

Improving GPU Energy Efficiency through an Application-transparent Frequency Scaling Policy with Performance Assurance.

View our paper at https://dl.acm.org/doi/abs/10.1145/3627703.3629584

## License
See the LICENSE file.
