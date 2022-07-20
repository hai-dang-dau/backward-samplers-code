=== Requirements ===

python>=3.8
numpy
scipy
numba
matplotlib
seaborn
statsmodels
pandas
git+https://github.com/nchopin/particles.git@c91d4b8ea05d1a4f4f148ac9cdf383c36eeb13a4

=== Program structure ===

The main.py files for the linear Gaussian and the Lotka-Volterra SDE examples are located respectively in ./program/brownian/main.py and ./programs/sde/main.py. They read an input csv file containing multiple run configurations and output the results in the JSON format, from which our matplotlib-based plots are drawn. Note that the input files are modified during execution: each time a configuration is read, it is removed from the corresponding csv file.

The main.py files point to the run_test.csv files by default. They contain small scale examples to verify that installation is successful. To reproduce data in the article, change the pointer to the run_full.csv files (warning: these files tell the program to execute using 50 cores).

Parallel executions are performed via the fork mechanism. As such, the programs might not function properly under Windows and macOS 10.13 or above.

=== Input file structure ===

Here is the content of the ./programs/sde/output/run_test.csv file:

seed_execution,nruns_per_core,ncores,maxmem,json_maxsize,config_file,N,ESSrmin,resampling,skeleton_converter,smooth_mode,algo,ancestor_coupling_mode,n_highlight,highlight_every
66776,2,2,3596,500,simple_start,100,1.0,systematic,NaN,intractable,intractable,Adjacent,1,1

Below is an explanation of the most important keys:
* seed_execution: the numpy seed used to run the configuration (important for reproducibility)
* nruns_per_core: number of runs per CPU core
* ncores: number of CPU cores that will be used
* maxmem: maximum allowed memory for each run (per CPU core) in MB. If the program uses more memory than this, the MemoryError exception will be raised
* json_maxsize: maximum JSON output file size
* config_file: file that contains the mathematical problem description. In this example, "simple_start" points to the "./programs/sde/config/simple_start.json" file which contains parameters for the Lotka-Volterra stochastic equations
* N: number of particles
* smooth_mode: either "online", "offline" or "intractable". In this example, the model transition density is unknown, so the intractable smoother is necessary
* ancestor_coupling_mode: either "Permutation" or "Adjacent". The "Adjacent" mode sorts resampled particles by the Hilbert curve, whereas the "Permutation" mode just leaves them in the random order. See Section 4.2.3 (Good ancestor couplings) and Algorithm 10 (The Adjacent Resampler) of the article

=== Output file structure ===

The main.py file reads each line of the input csv file, processes it and creates two output JSON files with names in the format AA_BB_input.json and AA_BB_output.json; where AA is a random prefix and BB is the date and time at which the line is read. The AA_BB_input.json file is essentially a copy of the processed line in the csv file, whereas the AA_BB_output.json file contains the actual result of the experiment. Here is an explanation of most important keys in the example file ./programs/sde/output/tGebSJ_2022-07-19 16-27-25.385134_output_0.json (recall that it corresponds to a small scale Lotka-Volterra SDE model with T = 5, smoothed using the intractable smoother described in Section 4.2 of the article):

* coupling_rate: a list of coupling rates at different times t (for the intractable smoother only)
* filtering_stability_diag: list of estimated filtering expectations at different times t. Used to verify that filtering is indeed stable (otherwise smoothing won't work, see Section 6.1 (Practical recommendation) of the article)
* exec_times: time spent by the intractable smoother at different t
* expectations: estimated expectations of the additive function at different t
* ESS_ratios: the ESS ratio at different t

Naturally, different smoothing algorithms will give rise to slightly different output JSON files.