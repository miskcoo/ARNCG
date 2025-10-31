from torch.optim import Adam, LBFGS, Optimizer
# from .nys_newton_cg_adap import NysNewtonCGAdap
from .adap_newton_cg import AdapNewtonCG

class Adam_LBFGS_NNCG_Adap(Optimizer):
    def __init__(self, params, switch_epoch1, switch_epoch2, precond_update_freq, adam_params, lbfgs_params, nncg_params):
        # defaults = dict(switch_epoch=switch_epoch, adam_params=adam_params, lbfgs_params=lbfgs_params)

        self.switch_epoch1 = switch_epoch1
        self.switch_epoch2 = switch_epoch2
        self.precond_update_freq = precond_update_freq
        self.params = list(params)
        self.adam = Adam(self.params, **adam_params)
        self.lbfgs = LBFGS(self.params, **lbfgs_params)
        # self.nncg = NysNewtonCGAdap(self.params, **nncg_params)
        self.nncg = AdapNewtonCG(self.params, **nncg_params)

        super(Adam_LBFGS_NNCG_Adap, self).__init__(self.params, defaults={})

        self.state['epoch'] = 0

    def step(self, closure=None):
        if self.state['epoch'] < self.switch_epoch1:
            self.adam.step(closure)
            self.state['epoch'] += 1

        elif self.state['epoch'] < self.switch_epoch2:
            if self.state['epoch'] == self.switch_epoch1:
                print(f'Switching to LBFGS optimizer at epoch {self.state["epoch"]}')
            self.lbfgs.step(closure)
            self.state['epoch'] += 1
            
        else:
            if self.state['epoch'] == self.switch_epoch2:
                print(f'Switching to AdapNewtonCG optimizer at epoch {self.state["epoch"]}')
            _, grad = self.nncg.step(closure)
            self.state['epoch'] += 1
            return grad

        
