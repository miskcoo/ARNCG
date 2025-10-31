import torch
from torch.optim import Optimizer
from torch.func import vmap
from functools import reduce
import math

def _apply_nys_precond_inv(U, S_mu_inv, mu, lambd_r, x):
    """Applies the inverse of the Nystrom approximation of the Hessian to a vector."""
    z = U.T @ x
    z = (lambd_r + mu) * (U @ (S_mu_inv * z)) + (x - U @ z)
    return z

def _capped_cg(G, reltol, abstol, rho, hessvec_fn, max_iters, preconditioner=None):

    if preconditioner is not None:
        lambd_r = preconditioner['S'][preconditioner['r'] - 1]
        S_mu_inv = (preconditioner['S'] + 2 * rho) ** (-1)
        def apply_precond_inv(x):
            return _apply_nys_precond_inv(preconditioner['U'], S_mu_inv, 2 * rho, lambd_r, x)


    Y = torch.zeros_like(G)
    Hy = torch.zeros_like(G) 
    R = G

    if preconditioner is not None: 
        with torch.no_grad():
            pR = apply_precond_inv(R)
            P = -pR.clone()
        Hr = hessvec_fn(pR)
        Hp = -Hr
    else:
        Hr = hessvec_fn(R)
        P = -R
        Hp = -Hr

    norm_rr0 = torch.dot(R, R)

    U = 0 
    kappa = (U + 2 * rho) / rho
    tau = math.sqrt(kappa) / (math.sqrt(kappa) + 1)
    T = 4 * kappa ** 4 / (1 - math.sqrt(kappa)) ** 2

    d_type = 'EMPTY'

    norm_rr = norm_rr0 
    norm_pbarHp = 0
    barHp = torch.zeros_like(G)

    for it in range(max_iters):

        if it > 0:
            with torch.no_grad():
                if preconditioner is not None:
                    dot_rpr = torch.dot(R, pR)
                    alpha = dot_rpr / norm_pbarHp
                else:
                    alpha = norm_rr / norm_pbarHp
                Y = Y + alpha * P
                Hy = Hy + alpha * Hp

                R = alpha * barHp + R

            if preconditioner is not None:
                with torch.no_grad():
                    pR = apply_precond_inv(R)
                Hr = hessvec_fn(pR)
                with torch.no_grad():
                    beta = torch.dot(R, pR) / dot_rpr
                    P = beta * P - pR
                    Hp = beta * Hp - Hr
            else:
                Hr = hessvec_fn(R)
                with torch.no_grad():
                    beta = torch.dot(R, R) / norm_rr
                    P = beta * P - R
                    Hp = beta * Hp - Hr

        with torch.no_grad():
            norm_Hp = torch.dot(Hp, Hp)
            norm_pp = torch.dot(P, P)

            barHp = Hp + 2 * rho * P
            norm_pbarHp = torch.dot(P, barHp)

            if math.sqrt(norm_Hp) > U * math.sqrt(norm_pp):
                U = math.sqrt(norm_Hp) / math.sqrt(norm_pp)
            norm_Hr = torch.dot(Hr, Hr)
            norm_rr = torch.dot(R, R)
            if preconditioner is not None:
                norm_prpr = torch.dot(pR, pR)
                if math.sqrt(norm_Hr) > U * math.sqrt(norm_prpr):
                    U = math.sqrt(norm_Hr) / math.sqrt(norm_prpr)
            else:
                if math.sqrt(norm_Hr) > U * math.sqrt(norm_rr):
                    U = math.sqrt(norm_Hr) / math.sqrt(norm_rr)
            norm_Hy = torch.dot(Hy, Hy)
            norm_yy = torch.dot(Y, Y)
            if math.sqrt(norm_Hy) > U * math.sqrt(norm_yy):
                U = math.sqrt(norm_Hy) / math.sqrt(norm_yy)

            kappa = (U + 2 * rho) / rho
            tau = math.sqrt(kappa) / (math.sqrt(kappa) + 1)
            T = 4 * kappa ** 4 / (1 - math.sqrt(kappa)) ** 2
            hat_xi = reltol / (3 * kappa)

            # negative curvature detection
            norm_ybarHy = torch.dot(Y, Hy + 2 * rho * Y)
        if norm_ybarHy < rho * norm_yy:
            Z = Y
            d_type = 'NC'
            break
        elif math.sqrt(norm_rr) < hat_xi * math.sqrt(norm_rr0) and math.sqrt(norm_rr) < abstol:
            Z = Y 
            d_type = 'SOL'
            break
        elif norm_pbarHp < rho * norm_pp:
            Z = P
            d_type = 'NC'
            break
        elif math.sqrt(norm_rr) > math.sqrt(T) * (tau ** (it * 0.5)) * math.sqrt(norm_rr0):

            ##### ========================== WARNING ==========================
            ##### this branch is not well-tested, since we never reached this condition in our experiments.
            ##### ========================== WARNING ==========================

            alpha = norm_rr / norm_pbarHp
            Y_extra = Y + alpha * P
            HY_extra = Hy + alpha * Hp

            Y = torch.zeros_like(G)
            Hy = torch.zeros_like(G)
            R = G  
            Hr = hessvec_fn(R)
            P = -R
            Hp = -Hr

            d_type = 'ERR'

            for _ in range(1, it + 2):

                diff = Y_extra - Y
                sqrnorm_diff = torch.dot(diff, diff)
                diff_H_diff = torch.dot(diff, HY_extra - Hy)
                if diff_H_diff < rho * sqrnorm_diff:
                    Z = diff
                    d_type = 'NC'
                    break

                if it > 0:
                    alpha = norm_rr / norm_pbarHp
                    Y = Y + alpha * P
                    Hy = Hy + alpha * Hp

                    R = alpha * barHp + R
                    Hr = hessvec_fn(R)

                    beta = torch.dot(R, R) / norm_rr
                    P = beta * P - R
                    Hp = beta * Hp - Hr

                norm_Hp = torch.dot(Hp, Hp)
                norm_pp = torch.dot(P, P)

                barHp = Hp + 2 * rho * P
                norm_pbarHp = torch.dot(P, barHp)

                norm_Hr = torch.dot(Hr, Hr)
                norm_rr = torch.dot(R, R)
                norm_Hy = torch.dot(Hy, Hy)
                norm_yy = torch.dot(Y, Y)

            break

    if d_type == 'EMPTY':
        Z = Y
        d_type = 'SOL'

    return d_type, Z, it, norm_rr


class AdapNewtonCG(Optimizer):
    def __init__(self, 
                 params, 
                 beta=0.5,
                 mu=0.3,
                 min_alpha=1e-4,
                 cg_reltol=1e-2,
                 cg_abstol=1e-2,
                 cg_maxiter=1000,
                 gamma=2,
                 tau_minus=0.3,
                 tau_plus=1.0,
                 theta=1.0,
                 chunk_size=1,
                 use_precond=True,
                 rank=60,
                 M_recycle_limit=1e10,
                 verbose=False,
                 **kwargs,
                 ):
        defaults = dict(
                beta=beta,
                mu=mu,
                min_alpha=min_alpha,
                cg_reltol=cg_reltol,
                cg_abstol=cg_abstol,
                cg_maxiter=cg_maxiter,
                gamma=gamma,
                tau_minus=tau_minus,
                tau_plus=tau_plus,
                theta=theta,
                use_precond=use_precond,
                rank=rank,
                chunk_size=chunk_size,
                M_recycle_limit=M_recycle_limit,
                verbose=verbose
                )

        self.n_iters = 0
        self.beta = beta
        self.mu = mu
        self.min_alpha = min_alpha
        self.cg_reltol = cg_reltol
        self.cg_abstol = cg_abstol
        self.cg_maxiter = int(cg_maxiter)
        self.gamma = gamma
        self.tau_minus = tau_minus
        self.tau_plus = tau_plus
        self.theta = theta
        self.verbose = verbose
        self.use_precond = use_precond
        self.rank = rank
        self.chunk_size = chunk_size
        self.M_recycle_limit = M_recycle_limit
        self.hvp_times_single_step = 0

        super(AdapNewtonCG, self).__init__(params, defaults)

        self._params = self.param_groups[0]['params']
        self._params_list = list(self._params)
        self._numel_cache = None

    def step(self, closure=None):
        self.hvp_times_single_step = 0

        if self.n_iters == 0:
            # Store the previous direction for warm starting PCG
            self.old_dir = torch.zeros(
                self._numel(), device=self._params[0].device)
            self.M = 1
            self.pending_M_update = False

        # NOTE: The closure must return both the loss and the gradient
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss, grad_tuple = closure()

        g = torch.cat([grad.view(-1) for grad in grad_tuple if grad is not None])

        norm_g = torch.norm(g)

        accel_factor = 1.0 
        if self.n_iters > 0:
            accel_factor = min(1., norm_g / self.prev_norm_g) ** self.theta
        self.prev_norm_g = norm_g

        if self.pending_M_update:
            predict_descent = self.predict_lipcoeff * min(self.omega ** 3, norm_g ** 2 / self.omega);
            if self.f_old < 4 / 33 * self.mu * predict_descent + loss:
                self.M *= self.gamma
            self.pending_M_update = False
            # release pinned memory
            self.predict_lipcoeff = None
            self.omega = None
            self.f_old = None

        # indeed, there is no need to set the upper bound of M 
        # Ideally, it will be adjusted automatically 
        # However, there may be some numerical issues (especially for float32 and the loss becomes small)
        # such that the updating rule of M becomes unreliable.
        # In our experiments, we use float64 and this condition is not triggered,
        # M is always less than M_max (1e4), but for float32, it may be triggered.
        # the this will help slighly reduce the loss at the end of iterations. 
        M_max = self.M_recycle_limit
        M = min(self.M, M_max) #self.M
        #print("M: ", M)

        if self.cg_maxiter is None or self.cg_maxiter == 0:
            self.cg_maxiter = torch.numel(g)

        # One step update
        for group_idx, group in enumerate(self.param_groups):
            def hvp_temp(x):
                self.hvp_times_single_step += 1
                return self._hvp(g, self._params_list, x)

            # Calculate the Newton direction
            omega_base = torch.sqrt(norm_g)
            omega = omega_base * accel_factor
            cur_regularizer = M ** 0.5 * omega
            cg_reltol = min(self.cg_reltol, cur_regularizer)
            d_type, tilde_d, cg_iters, cg_res = _capped_cg(
                    g, cg_reltol, self.cg_abstol, cur_regularizer, hvp_temp, self.cg_maxiter,
                    preconditioner={'U': self.U, 'S': self.S, 'r': self.rank} if self.use_precond else None)

            # Store the previous direction for warm starting PCG
            self.old_dir = tilde_d

            # Check if d is a descent direction
            # print("d_type: ", d_type)
            # print("dot product: ", torch.dot(tilde_d, g))

            # perform linesearch 
            def obj_func(x, t, dx):
                self._add_grad(t, dx)
                loss = closure()[0].item()
                self._set_param(x)
                return loss

            linesearch_failure = False

            if d_type == 'SOL':
                d = tilde_d
                x_init = self._clone_param()
                dot_g_d = torch.dot(g, d)
                alpha = 1.0
                while alpha > self.min_alpha:
                    f_new = obj_func(x_init, alpha, d)
                    if f_new <= loss + self.mu * alpha * dot_g_d:
                        break
                    alpha *= self.beta

                if alpha <= self.min_alpha:
                    linesearch_failure = True

                t = alpha
            else:  # d_type == 'NC' 
                norm_tilde_d = torch.norm(tilde_d) 
                normalized_tilde_d = tilde_d / norm_tilde_d
                L_d = torch.abs(torch.dot(normalized_tilde_d, hvp_temp(normalized_tilde_d))) / M
                d = -L_d * torch.sign(torch.dot(g, normalized_tilde_d)) * normalized_tilde_d
                norm_d = torch.norm(d)

                x_init = self._clone_param()

                alpha = 1.0
                while alpha > self.min_alpha:
                    f_new = obj_func(x_init, alpha, d)
                    if f_new <= loss - M * self.mu * (alpha ** 2) * (norm_d ** 3):
                        break
                    alpha *= self.beta

                if alpha <= self.min_alpha:
                    linesearch_failure = True

                t = alpha

            # update lipschitz constant
            predict_lipcoeff = self.tau_plus / M ** 0.5
            M_dec_coeff = self.mu * self.tau_minus

            if d_type == 'SOL':
                if alpha < 1:
                    predict_descent = predict_lipcoeff * omega ** 3
                    if loss < self.beta * self.mu * predict_descent + f_new:
                        self.M = M * self.gamma
                else:
                    self.predict_lipcoeff = predict_lipcoeff
                    self.omega = omega
                    self.pending_M_update = True
                    self.f_old = loss
                    M_dec_coeff = M_dec_coeff * 4 / 33
            else:
                predict_descent = predict_lipcoeff * omega ** 3
                if loss < ((1 - 2 * self.mu) * self.beta) ** 2 * self.mu * predict_descent + f_new:
                    self.M = M * self.gamma

            if loss > M_dec_coeff / (M ** 0.5) * (omega_base ** 3) + f_new:
                self.M = M / self.gamma 


            self.state[group_idx]['t'] = t

            print(f"iter: {self.n_iters}, loss: {loss}, norm_g: {norm_g}, M: {M}, t: {t}, cg_iters: {cg_iters}, cg_res: {cg_res}, cg_d_type: {d_type}, linesearch_failure: {linesearch_failure}")

            # update parameters
            if linesearch_failure and self.M < M_max:
                if self.verbose:
                    print('Linesearch failed')
                self.M = self.gamma * M
                self.pending_M_update = False
            else:
                # This branch WON'T be reached for float64 in our experiments 
                # Please check the comments above when setting M = min(self., self.M_recycle_limit)
                ls = 0
                for p in group['params']:
                    np = torch.numel(p)
                    dp = d[ls:ls+np].view(p.shape)
                    ls += np
                    p.data.add_(dp, alpha=t)
                if self.M >= M_max:
                    self.M = 1.

        self.n_iters += 1

        return loss, g

    def _hvp_vmap(self, grad_params, params):
        return vmap(lambda v: self._hvp(grad_params, params, v), in_dims=0, chunk_size=self.chunk_size)

    def _hvp(self, grad_params, params, v):
        Hv = torch.autograd.grad(grad_params, params, grad_outputs=v,
                                 retain_graph=True)
        Hv = tuple(Hvi.detach() for Hvi in Hv)
        return torch.cat([Hvi.reshape(-1) for Hvi in Hv])

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # Avoid in-place operation by creating a new tensor
            p.data = p.data.add(
                update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def update_preconditioner(self, grad_tuple):
        if not self.use_precond:
            return

        # Flatten and concatenate the gradients
        gradsH = torch.cat([gradient.view(-1)
                           for gradient in grad_tuple if gradient is not None])

        # Generate test matrix (NOTE: This is transposed test matrix)
        p = gradsH.shape[0]
        Phi = torch.randn(
            (self.rank, p), device=gradsH.device, dtype=gradsH.dtype) / (p ** 0.5)
        Phi = torch.linalg.qr(Phi.t(), mode='reduced')[0].t()

        Y = self._hvp_vmap(gradsH, self._params_list)(Phi)

        # Calculate shift
        shift = torch.finfo(Y.dtype).eps
        Y_shifted = Y + shift * Phi

        # Calculate Phi^T * H * Phi (w/ shift) for Cholesky
        choleskytarget = torch.mm(Y_shifted, Phi.t())

        # Perform Cholesky, if fails, do eigendecomposition
        # The new shift is the abs of smallest eigenvalue (negative) plus the original shift
        try:
            C = torch.linalg.cholesky(choleskytarget)
        except:
            # eigendecomposition, eigenvalues and eigenvector matrix
            try:
                eigs, eigvectors = torch.linalg.eigh(choleskytarget)
            except:
                return
            shift = shift + torch.abs(torch.min(eigs)) * 2
            # add shift to eigenvalues
            eigs = eigs + shift
            # put back the matrix for Cholesky by eigenvector * eigenvalues after shift * eigenvector^T
            try:
                C = torch.linalg.cholesky(
                    torch.mm(eigvectors, torch.mm(torch.diag(eigs), eigvectors.T)))
            except:
                # eigendecomposition failed, use previous preconditioner
                return

        try:
            B = torch.linalg.solve_triangular(
                C, Y_shifted, upper=False, left=True)
        # temporary fix for issue @ https://github.com/pytorch/pytorch/issues/97211
        except:
            B = torch.linalg.solve_triangular(C.to('cpu'), Y_shifted.to(
                'cpu'), upper=False, left=True).to(C.device)
            
        # B = V * S * U^T b/c we have been using transposed sketch
        _, S, UT = torch.linalg.svd(B, full_matrices=False)
        self.U = UT.t()
        self.S = torch.max(torch.square(S) - shift, torch.tensor(0.0))

        self.rho = self.S[-1]

        if self.verbose:
            print(f'Approximate eigenvalues = {self.S}')

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            # Replace the .data attribute of the tensor
            p.data = pdata.data
