function [x_opt, norm_g, records, hess_evals, grad_evals, func_evals] = AdapNewtonCG(x0, tol, options)

    % options.grad_fn, hessvec_fn, loss_fn

    tic;

    % (x, u, v) => u^T * v
    dot_fn = options.dot_fn;
    % (x, v) => ||v||
    norm_fn = options.norm_fn;
    % (x) => g
    grad_fn = options.grad_fn;
    if isfield(options, 'hessvec_fn')
        % (x, v) => H(x) * v
        hessvec_fn = options.hessvec_fn;
    end
    % (x) => f(x)
    loss_fn = options.loss_fn;
    verbose = options.verbose;

    max_iter = options.max_iter;
    max_omega = options.max_omega;  %% set it to inf, not used in our article

    % line search parameters
    beta = options.beta;
    mu = options.mu;
    min_alpha = options.min_alpha;  %% m_max = log_beta(min_alpha)

    cg_reltol = options.cg_reltol;
    cg_abstol = options.cg_abstol;
    cg_maxiter = options.cg_maxiter;   %% set it to "problem dimension" + 1

    % adaptive parameters
    gamma = options.gamma;
    tau_minus = options.tau_minus;
    tau_plus = options.tau_plus;

    fallback_enabled = options.fallback_enabled;
    if fallback_enabled
        fallback_growth_threshold = options.fallback_growth_threshold;  %% this is lambda^{-1}
        fallback_shrink_threshold = options.fallback_shrink_threshold;  %% this is lambda in appendix
    end

    theta = options.theta;

    grad_evals = 1;
    func_evals = 1;
    hess_evals = 0;
    rejected_step_flag = 0;

    % M0 = 1.0e-10;
    M = 1;
    x = x0;
    g = grad_fn(x);
    f = loss_fn(x);
    norm_g = norm_fn(x, g);
    norm_g_min = norm_g;
    prev_norm_g = norm_g;
    prev_norm_g_min = norm_g_min;

    fallback_flag = 0;

    records = [];

    for it = 1:max_iter

        if norm_g < tol
            break;
        end

        % this is the Hessian evaluation for current iteration
        % when the previous step is rejected, x_{k} = x_{k-1}
        hess_evals = hess_evals + (1 - rejected_step_flag);
        rejected_step_flag = 0;

        % setup omega_k^f for fallback step
        if strcmp(options.regularization_policy, 'gradient')
            omega_base = sqrt(norm_g);
        elseif strcmp(options.regularization_policy, 'minimum_gradient')
            omega_base = sqrt(norm_g_min);
        elseif strcmp(options.regularization_policy, 'fixed')
            omega_base = options.fixed_omega;
        else
            error('Unknown regularization policy');
        end

        % setup omega_k^t for trial step
        if ~fallback_flag
            if strcmp(options.acceleration_policy, 'gradient')
                omega = omega_base * min(1, (norm_g / prev_norm_g) ^ theta);
            elseif strcmp(options.acceleration_policy, 'minimum_gradient')
                omega = omega_base * (norm_g_min / prev_norm_g_min) ^ theta;
            elseif strcmp(options.acceleration_policy, 'fixed')
                omega = options.fixed_omega;
            else
                error('Unknown acceleration policy');
            end
        else
            % fallback step
            omega = omega_base;
            fallback_flag = 0;
        end

        % max_omega = inf in our article
        omega = min(omega, max_omega);

        cgtol = min(cg_reltol, sqrt(M) * omega);
        %cgtol = min(cg_reltol, omega);

        if isfield(options, 'hessvec_fn')
            current_hessvec_fn = @(v) hessvec_fn(x, v);
        else 
            current_hess = options.hess_fn(x);
            current_hessvec_fn = @(v) current_hess * v;
        end

        %% cg_policy = 'save' is not suggested, it takes many memory
        %% recommended: cg_policy = 'recompute'
        [tilde_d, cg_norm_rr, cg_it, cg_dtype] = CappedCG(...
            x, g, cgtol, cg_abstol, sqrt(M) * omega, ...
            current_hessvec_fn, ...
            @(u, v) dot_fn(x, u, v), ...
            cg_maxiter, strcmp(options.cg_policy, 'save'), verbose);

        dot_g_tilde_d = dot_fn(x, g, tilde_d);

        linesearh_failure_flag = 0;
        smaller_stepsize_flag = 0;

        if strcmp(cg_dtype, 'SOL')
            d = tilde_d;
            norm_d = norm_fn(x, d);
            dot_g_d = dot_g_tilde_d;

            % line search
            alpha = 1;
            while alpha > min_alpha
                x_new = x + alpha * d;
                f_new = loss_fn(x_new);
                func_evals = func_evals + 1;
                if f_new <= f + mu * alpha * dot_g_d
                    break;
                end
                alpha = alpha * beta;
            end

            % switch to smaller stepsize 
            hat_alpha = sqrt(omega / sqrt(M) / norm_d);
            if hat_alpha < 1 && alpha <= min_alpha
                alpha = 1;
                smaller_stepsize_flag = 1;
                while alpha > min_alpha
                    x_new = x + hat_alpha * alpha * d;
                    f_new = loss_fn(x_new);
                    func_evals = func_evals + 1;
                    if f_new <= f + mu * hat_alpha * alpha * dot_g_d
                        break;
                    end
                    alpha = alpha * beta;
                end

                if alpha <= min_alpha
                    linesearh_failure_flag = 1;
                end

                alpha = hat_alpha * alpha;
            elseif alpha <= min_alpha
                linesearh_failure_flag = 1;
            end
        else % NC
            norm_tilde_d = norm_fn(x, tilde_d);
            normalized_tilde_d = tilde_d / norm_tilde_d;
            L_d = abs(dot_fn(x, current_hessvec_fn(normalized_tilde_d), normalized_tilde_d)) / M;
            d = -L_d * sign(dot_fn(x, g, normalized_tilde_d)) * normalized_tilde_d;
            norm_d = norm_fn(x, d);

            dot_g_d = dot_fn(x, g, d);

            if verbose
                fprintf('Iter %d: NC, L_d = %e, dot_g_d = %e\n', it, L_d, dot_g_d);
            end

            % line search
            alpha = 1;
            while alpha > min_alpha
                x_new = x + alpha * d;
                f_new = loss_fn(x_new);
                func_evals = func_evals + 1;
                if f_new <= f - M * mu * (alpha ^ 2) * (norm_d ^ 3)
                    break;
                end
                alpha = alpha * beta;
            end

            if alpha <= min_alpha
                linesearh_failure_flag = 1;
            end
        end

        M_new = M;
        % x_new has been computed in line search
        % x_new = x + alpha * d;
        % loss has been computed in line search
        % f_new = loss_fn(x_new);

        % adaptive M
        predict_lipcoeff = tau_plus / sqrt(M);

        M_dec_coeff = mu * tau_minus;
        M_dec_omega = omega_base ^ 3;

        gradient_computed = 0;

        if strcmp(cg_dtype, 'SOL')
            if alpha < 1
                predict_descent = predict_lipcoeff * omega ^ 3;
                if f < beta * mu * predict_descent + f_new
                    M_new = M * gamma;
                end
            else
                g_new = grad_fn(x_new);  
                grad_evals = grad_evals + 1;
                % the gradient for x_new is already computed
                gradient_computed = 1;
                predict_descent = predict_lipcoeff * min(omega ^ 3, norm_fn(x, g_new) ^ 2 / omega);
                M_dec_coeff = M_dec_coeff * 4 / 33;
                if f < 4 / 33 * mu * predict_descent + f_new
                    M_new = M * gamma;
                end
            end
        else
            predict_descent = predict_lipcoeff * omega ^ 3;
            if f < ((1 - 2 * mu) * beta) ^ 2 * mu * predict_descent + f_new
                M_new = M * gamma;
            end
        end

        if f > M_dec_coeff * predict_lipcoeff * (omega_base ^ 3) + f_new
            M_new = M / gamma; 
        end

        Delta = f - f_new;
        ratio = Delta / omega_base ^ 3 * sqrt(M);
        if verbose
            fprintf('Iter %d: f = %e, |g| = %e, |d| = %e, alpha = %e, M = %e, omega = %e, cg_it = %d, cg_res = %e, Delta = %e, ratio = %e, toc = %f\n', ...
                it, f, norm_g, norm_d, alpha, M, omega, cg_it, cg_norm_rr, Delta, ratio, toc);
        end

        if linesearh_failure_flag
            if verbose
                fprintf('  [W] Iter %d: alpha <= min_alpha, the step is rejected.\n', it);
            end
            %% an underestimation of the Lipschitz constant
            %% reject the step, and increase M
            M_new = M * gamma;
            x_new = x;
            f_new = f;
            g_new = g;
            rejected_step_flag = 1;
            gradient_computed = 1;
        end

        %% the fallback step for theta = 0 is equivalent to the trial step 
        %% and removing it does not affect the complexity, so we skip it 
        if fallback_enabled && theta > 0
            % if current step is the fallback step, then prev_norm_g = norm_g 
            % so we won't get consecutive fallback steps
            if norm_g < prev_norm_g * fallback_shrink_threshold
                if gradient_computed == 0
                    g_new = grad_fn(x_new);
                    grad_evals = grad_evals + 1;
                    gradient_computed = 1;
                end
                next_norm_g = norm_fn(x_new, g_new);
                if next_norm_g > norm_g * fallback_growth_threshold
                    if verbose
                        fprintf('  [W] Iter %d: skipped, norm_g / prev_norm_g = %e, next_norm_g / norm_g = %e\n', ...
                            it, norm_g / prev_norm_g, next_norm_g / norm_g);
                    end
                    fallback_flag = 1;
                    rejected_step_flag = 1;
                    x_new = x;
                    g_new = g;
                    f_new = f;
                    M_new = M;
                end
            end
        end

        record_item = struct();
        record_item.iter = it;
        record_item.f = f;
        record_item.norm_g = norm_g; % this is the previous norm_g
        record_item.norm_d = norm_d;
        record_item.alpha = alpha;
        record_item.M = M;
        record_item.cg_it = cg_it;   %% number of CG iterations at CURRENT ITERATION
        record_item.cg_res = cg_norm_rr;
        record_item.cg_dtype = cg_dtype;
        record_item.omega = omega;
        record_item.Delta = Delta;
        record_item.ratio = ratio;
        record_item.toc = toc;
        record_item.linesearch_failure_flag = linesearh_failure_flag;
        record_item.smaller_stepsize_flag = smaller_stepsize_flag;
        record_item.fallback_flag = fallback_flag;
        record_item.grad_evals = grad_evals;  %% ACCUMULATED gradient evaluations
        record_item.func_evals = func_evals;  %% ACCUMULATED function evaluations
        record_item.hess_evals = hess_evals;  %% ACCUMULATED Hessian evaluations
        records = [records, record_item];

        %% stopping criteria, when f and g are unchanged for many iterations
        if isfield(options, 'exit_for_many_unchanged_f_and_g')
            unchanged_f_and_g_len = options.exit_for_many_unchanged_f_and_g;
            if length(records) > unchanged_f_and_g_len
                if records(end).f == records(end - unchanged_f_and_g_len).f && ...
                    records(end).norm_g == records(end - unchanged_f_and_g_len).norm_g
                    fprintf(' [E] Iter %d: unchanged f and g for %d iterations\n', it, unchanged_f_and_g_len);
                    break;
                end
            end
        end

        x = x_new;
        % update the Lipschitz constant estimate
        M = M_new;

        % the gradient may have been computed in the line search
        % or when the step is rejected 
        if gradient_computed == 0
            grad_evals = grad_evals + 1;
            g = grad_fn(x);
        else
            g = g_new;
        end

        prev_norm_g = norm_g;
        prev_norm_g_min = norm_g_min;
        norm_g = norm_fn(x, g);
        norm_g_min = min(norm_g, norm_g_min);

        % the previous step costs some times
        records(end).toc = toc;
        records(end).grad_evals = grad_evals;

        % f = loss_fn(x);
        f = f_new;

        %% numerical issues
        if isnan(f) 
            fprintf(' [E] Iter %d: f is NaN\n', it);
            break;
        end

        %% possible numerical issues, you can remove this if needed
        if M > 1.0e40
            fprintf(' [E] Iter %d: M > 1.0e40\n', it);
            break;
        end

        if record_item.toc > options.max_time
            fprintf(' [E] Iter %d: max_time exceeded\n', it);
            break;
        end

        %% possible numerical issues
        %% but if you want to observe the local convergence for high accuracy, 
        %% you MUST remove this, otherwise the algorithm will stop early
        if isfield(options, 'minimal_norm_d') && norm_d < options.minimal_norm_d
            fprintf(' [E] Iter %d: norm_d < minimal_norm_d\n', it);
            break;
        end
    end

    if verbose
        fprintf('Iter %d: f = %e, |g| = %e\n', ...
            it, f, norm_fn(x, g));
    end

    x_opt = x;
%    norm_g = norm_fn(x, grad_fn(x));

    if verbose
        if isempty(records)
            %%% initial point is already optimal
            fprintf('initial point is already optimal\n');
        else
            fprintf('Total time = %f, |g| = %e, hesseval = %d, gradeval = %d, funceval = %d, hessvec = %d\n', ... 
                records(end).toc, norm_g, hess_evals, grad_evals, func_evals, sum([ records.cg_it ]));
        end
    end
    
end
