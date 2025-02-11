ARNCG
=====

Code for the following article:

```bibtex
@misc{zhou2025regularizednewtonmethodnonconvex,
      title={A Regularized Newton Method for Nonconvex Optimization with Global and Local Complexity Guarantees}, 
      author={Yuhao Zhou and Jintao Xu and Chenglong Bao and Chao Ding and Jun Zhu},
      year={2025},
      eprint={2502.04799},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2502.04799}, 
}
```

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

To use the algorithm in a general setting, begin by setting up a configuration structure (a MATLAB `struct`) that specifies various options. For a complete example, see the file [`matlab/TestCUTEst.m`](matlab/TestCUTEst.m). Below is a simple example configuration; refer to `matlab/TestCUTEst.m` for additional choices and explanations:

```matlab
% Set the problem dimension based on the initial point x0
dim = size(x0, 1);

% Initialize the options structure
options = struct();

% Maximum number of iterations
options.max_iter = 100000;

% Parameters for the Conjugate Gradient (CG) method:
%   - cg_reltol: Relative tolerance (\eta in Algorithm 1)
%   - cg_abstol: Absolute error tolerance for the CG solution (see Appendix)
%   - cg_maxiter: Maximum iterations for CG (ideally, the exact solution is found in "dim" iterations)
options.cg_reltol  = 0.01;
options.cg_abstol  = 0.01;
options.cg_maxiter = dim + 2;

% CG policy: 'recompute' indicates that CG history is not saved
options.cg_policy = 'recompute';

% \rho_k = min(max_omega, sqrt(M_k) * omega)
% Setting max_omega to inf reduces the method to the original ARNCG.
options.max_omega = inf;

% Maximum allowed time (in seconds); here set to 5 hours
options.max_time = 3600 * 5;

% Line search parameters
options.beta = 0.5;
options.mu   = 0.3;

% min_alpha is chosen so that m_max = log_beta(min_alpha), which is equivalent to m_max = 1
options.min_alpha = 0.3;

% Parameters for adaptivity
options.gamma     = 5;
options.tau_minus = 0.3;
options.tau_plus  = 1.0;

% Minimal norm threshold:
%   - To observe local convergence with high accuracy, either remove this parameter
%     or set it to a very small value.
options.minimal_norm_d = 2e-16;

% Verbosity:
%   - Set to a nonzero value to output iteration information
options.verbose = 0;

% Termination criterion:
%   - Exit if the function value and gradient do not change for many iterations
options.exit_for_many_unchanged_f_and_g = 20;

% Fallback mechanism:
%   - Disabling the fallback mechanism (i.e., setting lambda = 0) by default
options.fallback_enabled = 0;

% Regularization and acceleration policies:
%   - 'gradient' corresponds to the first regularization policy (ARNCG_g)
options.acceleration_policy  = 'gradient';
options.regularization_policy = 'gradient';

% Additional algorithm parameter
options.theta = 1;

% Initialize inner product and norm functions:
options.dot_fn  = @(x, u, v) sum(u .* v);
options.norm_fn = @(x, v) norm(v);

% The following function handles need to be defined according to your specific problem:
%   - Loss function:     (x) -> f(x)
%     options.loss_fn = @(x) ... ;
%
%   - Gradient function: (x) -> ∇f(x)
%     options.grad_fn = @(x) ... ;
%
%   - Hessian-vector product: (x, v) -> ∇²f(x)*v
%     options.hessvec_fn = @(x, v) ... ;
% Note: If multiple calls to `hessvec_fn` at the same point `x` require a
%       preprocessing step that is independent of `v`, you should perform this
%       preprocessing once and cache the results for efficiency.
```

Once your configuration (stored in the variable `options`) is ready, execute the following code:

```matlab
% 'options' is your configuration struct
[x_opt, norm_g, records, hess_evals, grad_evals, func_evals] = AdapNewtonCG(p.x0, 1.0e-5, options);
```

The function returns:

1. **x_opt**: The computed solution.
2. **norm_g**: The norm of the gradient at `x_opt`. Compare this value with the target tolerance (1.0e-5 in this example) to determine if the algorithm converged successfully.
3. **records**: Detailed information about the algorithm's progress.
4. **hess_evals**, **grad_evals**, and **func_evals**: The number of the Hessian, gradient, and function oracles accesses, respectively.
