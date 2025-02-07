    
results_directory = 'results.cutest.arncg';
% CopyCode;

problist = [ {'ARGLINA'} {'ARGLINB'} {'ARGLINC'} {'ARGTRIGLS'} {'ARWHEAD'} {'BA-L16LS'} {'BA-L21LS'} {'BA-L49LS'} {'BA-L73LS'} {'BDQRTIC'} {'BOX'} {'BOXPOWER'} {'BROWNAL'} {'BROYDN3DLS'} {'BROYDN7D'} {'BROYDNBDLS'} {'BRYBND'} {'CHAINWOO'} {'COATING'} {'COSINE'} {'CRAGGLVY'} {'CURLY10'} {'CURLY20'} {'CURLY30'} {'CYCLIC3LS'} {'CYCLOOCFLS'} {'DIXMAANA1'} {'DIXMAANB'} {'DIXMAANC'} {'DIXMAAND'} {'DIXMAANE1'} {'DIXMAANF'} {'DIXMAANG'} {'DIXMAANH'} {'DIXMAANI1'} {'DIXMAANJ'} {'DIXMAANK'} {'DIXMAANL'} {'DIXMAANM1'} {'DIXMAANN'} {'DIXMAANO'} {'DIXMAANP'} {'DIXON3DQ'} {'DQDRTIC'} {'DQRTIC'} {'EDENSCH'} {'EG2'} {'EIGENALS'} {'EIGENBLS'} {'EIGENCLS'} {'ENGVAL1'} {'EXTROSNB'} {'FLETBV3M'} {'FLETCBV2'} {'FLETCBV3'} {'FLETCHBV'} {'FLETCHCR'} {'FMINSRF2'} {'FMINSURF'} {'FREUROTH'} {'GENHUMPS'} {'GENROSE'} {'INDEF'} {'INDEFM'} {'INTEQNELS'} {'JIMACK'} {'KSSLS'} {'LIARWHD'} {'LUKSAN11LS'} {'LUKSAN15LS'} {'LUKSAN16LS'} {'LUKSAN17LS'} {'LUKSAN21LS'} {'LUKSAN22LS'} {'MANCINO'} {'MNISTS0LS'} {'MNISTS5LS'} {'MODBEALE'} {'MOREBV'} {'MSQRTALS'} {'MSQRTBLS'} {'NCB20'} {'NCB20B'} {'NONCVXU2'} {'NONCVXUN'} {'NONDIA'} {'NONDQUAR'} {'NONMSQRT'} {'OSCIGRAD'} {'OSCIPATH'} {'PENALTY1'} {'PENALTY2'} {'PENALTY3'} {'POWELLSG'} {'POWER'} {'QING'} {'QUARTC'} {'SBRYBND'} {'SCHMVETT'} {'SCOSINE'} {'SCURLY10'} {'SCURLY20'} {'SCURLY30'} {'SENSORS'} {'SINQUAD'} {'SPARSINE'} {'SPARSQUR'} {'SPIN2LS'} {'SPINLS'} {'SPMSRTLS'} {'SROSENBR'} {'SSBRYBND'} {'SSCOSINE'} {'TESTQUAD'} {'TOINTGSS'} {'TQUARTIC'} {'TRIDIA'} {'VARDIM'} {'VAREIGVL'} {'WOODS'} {'YATP1CLS'} {'YATP1LS'} {'YATP2CLS'} {'YATP2LS'} ];

% our method cannot solve these problems, so we skip them and mark them as failed
skiplist = [ {'FLETCHBV'} {'SCURLY10'} {'CYCLOOCFLS'} {'FLETCBV3'} {'BA-L49LS'} {'SCURLY20'} {'BA-L16LS'} {'BA-L21LS'} {'NONMSQRT'} ];

rand_suffix = index;
rand_suffix = num2str(rand_suffix);

% toggle these two lines to enable the diary
% dfile = [ 'cutest_', datestr(now, 'yyyy-mm-dd-HHMMSS'), '.', rand_suffix, '.log'];
% diary(dfile);

for ip = ip_range
    pname = problist{ip};

    if ismember(pname, skiplist)
        continue;
    end

    p = macup(pname);  % make a CUTEst problem

    fprintf('\n%d. Try %s:\n', ip, pname);

    % initialize the options
    dim = size(p.x0, 1);
    options = struct();
    options.max_iter = 100000;

    % \eta in Algorithm 1
    options.cg_reltol = 0.01;
    % the absolute error for CG solution (see Appendix)
    options.cg_abstol = 0.01;
    % ideally, CG finds the exact solution in "dim" iterations
    options.cg_maxiter = dim + 2;

    % \lambda^{-1} in Appendix (only used for options.fallback_enabled = 1)
    options.fallback_growth_threshold = 100;  % lambda = 0.01

    % \lambda in Appendix (only used for options.fallback_enabled = 1)
    options.fallback_shrink_threshold = 0.01;  % lambda = 0.01

    % do not save the CG history
    options.cg_policy = 'recompute';

    % \rho_k = \min(max_omega, \sqrt{M_k} \omega)
    % setting it to inf reduces to the original ARNCG
    options.max_omega = inf;

    % 5 hours
    options.max_time = 3600 * 5;

    % the linesearch parameters
    options.beta = 0.5;
    options.mu = 0.3;

    % m_max = log_beta(min_alpha),
    % this is equivalent to m_max = 1
    options.min_alpha = 0.3;

    % the parameters for the adaptive regularization
    options.gamma = 5;
    options.tau_minus = 0.3;
    options.tau_plus = 1.0;

    % If you want to observe the local convergence for high accuracy, 
    % you MUST remove this, or set it to a very small value
    options.minimal_norm_d = 2e-16;

    % output the information in each iteration
    options.verbose = 0;

    % if the function value and the gradient does not change for many iterations, we exit 
    options.exit_for_many_unchanged_f_and_g = 20;

    %% index = 0:  ARNCG_epsilon, lambda = 0.00, theta = 1
    %% index = 1:  ARNCG_epsilon, lambda = 0.00, theta = 0.5
    %% index = 2:  ARNCG_g,       lambda = 0.00, theta = 1
    %% index = 3:  ARNCG_g,       lambda = 0.00, theta = 0.5
    %% index = 4:  fixed omega
    %% index = 5:  ARNCG_epsilon, lambda = 0.00, theta = 0
    %% index = 6:  [unused]
    %% index = 7:  ARNCG_g,       lambda = 0.00, theta = 0

    %% index = 8:  ARNCG_epsilon, lambda = 0.01, theta = 1
    %% index = 9:  ARNCG_epsilon, lambda = 0.01, theta = 0.5
    %% index = 10: ARNCG_g,       lambda = 0.01, theta = 1
    %% index = 11: ARNCG_g,       lambda = 0.01, theta = 0.5
    %% index = 12: ARNCG_epsilon, lambda = 1.00, theta = 1
    %% index = 13: ARNCG_epsilon, lambda = 1.00, theta = 0.5
    %% index = 14: ARNCG_g,       lambda = 1.00, theta = 1
    %% index = 15: ARNCG_g,       lambda = 1.00, theta = 0.5

    if bitand(index, 8)
        % enable the fallback mechanism
        % you should also set the following two parameters
        %     options.fallback_growth_threshold
        %     options.fallback_shrink_threshold
        options.fallback_enabled = 1;
    else
        % disable the fallback mechanism (lambda = 0)
        options.fallback_enabled = 0;
    end

    if bitand(index, 2)
        % use the first regularization policy (ARNCG_g)
        options.acceleration_policy = 'gradient';
        options.regularization_policy = 'gradient';
    else
        % use the second regularization policy (ARNCG_\epsilon)
        options.acceleration_policy = 'minimum_gradient';
        options.regularization_policy = 'minimum_gradient';
    end

    if bitand(index, 4)
        if options.fallback_enabled
            options.fallback_growth_threshold = 1;  % lambda = 1
            options.fallback_shrink_threshold = 1;  % lambda = 1
            if bitand(index, 1)
                options.theta = 0.5;
            else
                options.theta = 1;
            end
        else
            if bitand(index, 1)
                options.theta = 0;
            else
                % rho_k = \sqrt{M_k} * options.fixed_omega
                options.acceleration_policy = 'fixed';
                options.regularization_policy = 'fixed';
                options.theta = 0;
                options.fixed_omega = sqrt(1.0e-5);
                if bitand(index, 2)
                    % this index is redundant
                    continue;
                end
            end
        end
    else
        if bitand(index, 1)
            options.theta = 0.5;
        else
            options.theta = 1;
        end
    end

    % initialize the functions
    options.dot_fn = @(x, u, v) sum(u .* v);
    options.norm_fn = @(x, v) norm(v);
    options.loss_fn = @p.objective;
    options.grad_fn = @cutest_grad;
    options.hessvec_fn = @cutest_hprod;

    % options

    [x_opt, norm_g, records, hess_evals, grad_evals, func_evals] = AdapNewtonCG(p.x0, 1.0e-5, options);

    decup(p);  % destroy the CUTEst problem

    fprintf('\n');
    if norm_g > 1.0e-5
        state = 'failed';
    else
        state = 'success';
    end

    if isempty(records)
        %%% initial point is already optimal
        fprintf('%s [%s]: time = %f, |g| = %e, hesseval = %d, gradeval = %d, funceval = %d, hessvec = %d\n', ... 
            pname, state, 0, norm_g, hess_evals, grad_evals, func_evals, 0);
    else
        fprintf('%s [%s]: time = %f, |g| = %e, hesseval = %d, gradeval = %d, funceval = %d, hessvec = %d\n', ... 
            pname, state, records(end).toc, norm_g, hess_evals, grad_evals, func_evals, sum([ records.cg_it ]));
    end

    % save the results
    % rand_suffix = round((rand() + 1) * 10000);
    if ~exist([ results_directory, '/', pname ], 'dir')
        mkdir([ results_directory, '/', pname ]);
    end

    save_data = struct(...
        'x_opt', x_opt, ...
        'norm_g', norm_g, ...
        'records', records, ...
        'options', options, ...
        'hess_evals', hess_evals, ...
        'grad_evals', grad_evals, ...
        'func_evals', func_evals, ...
        'pname', pname, ...
        'dim', dim ...
        );

    save([ results_directory, '/', pname, '/', datestr(now, 'yyyy-mm-dd-HHMMSS'), '.', rand_suffix, '.mat' ], ...
        '-fromstruct', save_data);
    % 
    % 
end
