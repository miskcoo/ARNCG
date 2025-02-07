
% cg_poliy = 'save' or 'recompute';
function [Z, norm_res, it, d_type] = CappedCG(X, G, reltol, abstol, rho, hessvec_fn, dot_fn, max_iter, save_history, verbose)

    Y = zeros(size(G));
    Hy = zeros(size(G));
    R = G;  % residual
    Hr = hessvec_fn(R);
    P = -R;
    Hp = hessvec_fn(P);

    norm_rr0 = dot_fn(R, R);

    U = 0;
    kappa = (U + 2 * rho) / rho;
    tau = sqrt(kappa) / (sqrt(kappa) + 1);
    T = 4 * kappa ^ 4 / (1 - sqrt(kappa)) ^ 2;

    d_type = 'EMPTY';

    for it = 0:max_iter

        % standard CG iteration
        if it > 0
            alpha = norm_rr / norm_pbarHp;
            Y = Y + alpha * P;  
            Hy = Hy + alpha * Hp;

            R = alpha * barHp + R;
            Hr = hessvec_fn(R);

            beta = dot_fn(R, R) / norm_rr;
            P = beta * P - R;
            Hp = beta * Hp - Hr;
        end

        if save_history
            Y_history(:, :, it + 1) = Y;
            Hy_history(:, :, it + 1) = Hy;
        end

        % update the norm of Hessian
        norm_Hp = dot_fn(Hp, Hp);
        norm_pp = dot_fn(P, P);

        barHp = Hp + 2 * rho * P;
        norm_pbarHp = dot_fn(P, barHp);

        if sqrt(norm_Hp) > U * sqrt(norm_pp)
            U = sqrt(norm_Hp) / sqrt(norm_pp);
        end
        norm_Hr = dot_fn(Hr, Hr);
        norm_rr = dot_fn(R, R);
        if sqrt(norm_Hr) > U * sqrt(norm_rr)
            U = sqrt(norm_Hr) / sqrt(norm_rr);
        end
        norm_Hy = dot_fn(Hy, Hy);
        norm_yy = dot_fn(Y, Y);
        if sqrt(norm_Hy) > U * sqrt(norm_yy)
            U = sqrt(norm_Hy) / sqrt(norm_yy);
        end

        kappa = (U + 2 * rho) / rho;
        tau = sqrt(kappa) / (sqrt(kappa) + 1);
        T = 4 * kappa ^ 4 / (1 - sqrt(kappa)) ^ 2;
        hat_xi = reltol / (3 * kappa);

        % negative curvature detection
        norm_ybarHy = dot_fn(Y, Hy + 2 * rho * Y);
        if norm_ybarHy < rho * norm_yy
            Z = Y; d_type = 'NC';
            if verbose
                fprintf(' Capped CG: NC at Y\n');
            end
            break;
%        elseif sqrt(norm_rr) < tol  % TODO: update to the relerr
        elseif sqrt(norm_rr) < hat_xi * sqrt(norm_rr0) && sqrt(norm_rr) < abstol
            Z = Y; d_type = 'SOL';
            break;
        elseif norm_pbarHp < rho * norm_pp 
            Z = P; d_type = 'NC';
            if verbose
                fprintf(' Capped CG: NC at P\n');
            end
            break;
        elseif sqrt(norm_rr) > sqrt(T) * (tau ^ (it / 2)) * sqrt(norm_rr0)
            % Insufficient descent

            if save_history
                alpha = norm_rr / norm_pbarHp;
                Y_history(:, :, it + 2) = Y + alpha * P;
                Hy_history(:, :, it + 2) = Hy + alpha * Hp;

                diff_Y = diff(Y_history, 1, 3);
                diff_Hy = diff(Hy_history, 1, 3);
                norm_diffY_barH_diffY = dot_fn(diff_Y, diff_Hy + 2 * rho * diff_Y);
                norm_diffY_diffY = dot_fn(diff_Y, diff_Y);
                d_type = 'ERR';
                for i = 1:it 
                    if norm_diffY_barH_diffY < rho * norm_diffY_diffY 
                        Z = diff_Y(:, :, i); d_type = 'NC';
                        break;
                    end
                end
            else
                alpha = norm_rr / norm_pbarHp;
                Y_extra = Y + alpha * P;
                HY_extra = Hy + alpha * Hp;

                Y = zeros(size(G));
                Hy = zeros(size(G));
                R = G;  % residual
                Hr = hessvec_fn(R);
                P = -R;
                Hp = hessvec_fn(P);

                d_type = 'ERR';

                for i = 1:it + 1

                    diff = Y_extra - Y;
                    sqrnorm_diff = dot_fn(diff, diff);
                    diff_H_diff = dot_fn(diff, HY_extra - Hy);
                    if diff_H_diff < rho * sqrnorm_diff
                        Z = diff; d_type = 'NC';
                        break;
                    end

                    if it > 0
                        alpha = norm_rr / norm_pbarHp;
                        Y = Y + alpha * P;  
                        Hy = Hy + alpha * Hp;

                        R = alpha * barHp + R;
                        Hr = hessvec_fn(R);

                        beta = dot_fn(R, R) / norm_rr;
                        P = beta * P - R;
                        Hp = beta * Hp - Hr;
                    end

                    % update the norm of Hessian
                    norm_Hp = dot_fn(Hp, Hp);
                    norm_pp = dot_fn(P, P);

                    barHp = Hp + 2 * rho * P;
                    norm_pbarHp = dot_fn(P, barHp);

                    norm_Hr = dot_fn(Hr, Hr);
                    norm_rr = dot_fn(R, R);

                    norm_Hy = dot_fn(Hy, Hy);
                    norm_yy = dot_fn(Y, Y);
                end
            end

            if verbose
                if d_type == 'NC'
                    fprintf(' Capped CG: NC with insufficient descent, %d / %d\n', i, it);
                else 
                    fprintf(' Capped CG: NC with insufficient descent, [ERROR: direction not found]\n');
                end
            end

            break;
        end
    end

    if strcmp(d_type, 'EMPTY')
        Z = Y; d_type = 'SOL';
    end

    norm_res = sqrt(norm_rr);
end

