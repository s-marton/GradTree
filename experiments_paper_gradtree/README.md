# GradTree: Gradient Based Decision Trees

This repository provides the code for the paper ***"GradTree: Learning Axis-Aligned Decision Trees with Gradient Descent"*** by Sascha Marton, Stefan Lüdtke, Christian Bartelt and Heiner Stuckenschmidt available under: https://arxiv.org/abs/2305.03515

The required versions of all packages are listed in "*relevantdependencies.txt*" and can be installed running "*install_requirements.sh*". We further provide the following files to reproduce the results from the paper:
* *GDTs-BINARY-HPO.ipynb*: Load the optimized parameters and generate results for all binary classification datasets from paper.
* *GDTs-MULTI-HPO.ipynb*: Load the optimized parameters and generate results for all multi-class classification datasets from paper.
* *GDTs-BINARY-DEFAULT.ipynb*: Use defualt parameters for all approaches and generate results for all binary classification datasets from paper.
* *GDTs-MULTI-DEFAULT.ipynb*: Use defualt parameters for all approaches and generate results for all binary classification datasets from paper.
* *GDTs-SINGLE.ipynb*: Standalone notebook for comparing the different approaches on a single trial for a specified dataset. This notebook provides the best overview on how to use our approach (GDTs).

By default, the GPU is used for the calculations of GDT and DNDT. This can be changed in the config ("*usegpu*"). If you receive a memory error for GPU computations, this is most likely due to the parallelization of the experiments on a single GPU. Please try to reduce the number of parallel computations ("*njobs*") or run on CPU (set "*usegpu*" to "*False*"). Please note that the single-GPU parallelization reduces the overall runtime of the experiments, but negsatively influences the individual runtimes. Therefore the runtimes noted in the paper are generated in a separate setting without single-GPU parallelization of multiple trials/datasets.

We further want to note that it is unfortunately not possible to generate reproducible results on a GPU with TensorFlow. However this only results in minor deviations of the performance and does not influence the results shown in the paper.

The Python version used in the experiments was Python 3.11.3.

The code was tested with a Linux distribution. For other distributions it might not work as intended. If there occor errors when running the "*install_requirements.sh*", please consider installing the packages manually. Thereby, it is important to follow the sequence of "*relevant_dependencies.txt*".
