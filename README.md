ARNCG
=====

## Usage

### Running the CUTEst Benchmark

To run the CUTEst benchmark, follow these steps:

1. First, install the [MatCUTEst](https://github.com/matcutest/matcutest_compiled) library.

2. Once MatCUTEst is installed, change your working directory to the `matlab` folder.

3. Use the following MATLAB code to set up and start the benchmark:

   ```matlab
   % Specify the range of test problems (124 available problems, indexed 1 to 124)
   ip_range = 1:124;

   % Select the method to test. There are 15 available choices (indices 0 to 15, excluding 6).
   % Below is a list of each index and its corresponding configuration:
   %   index = 0:  ARNCG_epsilon, lambda = 0.00, theta = 1
   %   index = 1:  ARNCG_epsilon, lambda = 0.00, theta = 0.5
   %   index = 2:  ARNCG_g,       lambda = 0.00, theta = 1
   %   index = 3:  ARNCG_g,       lambda = 0.00, theta = 0.5
   %   index = 4:  fixed omega
   %   index = 5:  ARNCG_epsilon, lambda = 0.00, theta = 0
   %   index = 6:  [unused]
   %   index = 7:  ARNCG_g,       lambda = 0.00, theta = 0
   %   index = 8:  ARNCG_epsilon, lambda = 0.01, theta = 1
   %   index = 9:  ARNCG_epsilon, lambda = 0.01, theta = 0.5
   %   index = 10: ARNCG_g,       lambda = 0.01, theta = 1
   %   index = 11: ARNCG_g,       lambda = 0.01, theta = 0.5
   %   index = 12: ARNCG_epsilon, lambda = 1.00, theta = 1
   %   index = 13: ARNCG_epsilon, lambda = 1.00, theta = 0.5
   %   index = 14: ARNCG_g,       lambda = 1.00, theta = 1
   %   index = 15: ARNCG_g,       lambda = 1.00, theta = 0.5
   index = 2;

   % Start the benchmark tests
   TestCUTEst;
   ```

### General Usage

To use the algorithm in a general setting, start by setting up the configuration. This configuration is a MATLAB struct that includes several fields; for an example, see `matlab/TestCUTEst.m`.

Once your configuration (stored in the variable `options`) is ready, execute the following code:

```matlab
% 'options' is your configuration struct
[x_opt, norm_g, records, hess_evals, grad_evals, func_evals] = AdapNewtonCG(p.x0, 1.0e-5, options);
```

The function returns:

1. **x_opt**: The computed solution.
2. **norm_g**: The norm of the gradient at `x_opt`. Compare this value with the target tolerance (1.0e-5 in this example) to determine if the algorithm converged successfully.
3. **records**: Detailed information about the algorithm's progress.
4. **hess_evals**, **grad_evals**, and **func_evals**: The number of times the Hessian, gradient, and function oracles were accessed, respectively.
