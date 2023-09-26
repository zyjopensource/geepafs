# CUDA Samples
All of the files in this folder are from NVIDIA CUDA Code Samples and are copied here only to facilitate the running of testing experiments.

To compile an application, go into a certain folder and run "make". This step should be done for each application.

We have made minor changes to the application source codes to extend their execution time to be several minutes. The execution time of the original codes are within a few seconds, which is not suitable for testing our GPU frequency scaling policy.

# Note of our changes
The specific changes we made include:
- for bandwidthTest, we increase MEMCOPY_ITERATIONS.
- for BlackScholes, we increase NUM_ITERATIONS in file BlackScholes.cu.
- for convolutionFFT2D, in file main.cpp, only the test2 part is in use. The comparison with CPU computation is removed. We also add a loop to repeat the main computation.
- for cudaTensorCoreGemm, we add a loop to repeat the main computation.
- for fastWalshTransform, we remove the comparison with CPU computation, and we add a loop to repeat the main computation.
- for reductionMultiBlockCG, we increase testIterations.
- for sortingNetworks, we increase numIterations in file main.cpp.
- for transpose, we increase NUM_REPS.

For all applications, we also add two timestamps to mark the start and the end of the applications for post-processing.

