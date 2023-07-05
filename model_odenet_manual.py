import torch
from torch import nn, autograd
import math
import numpy as np
from scipy.integrate import odeint, solve_ivp
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)

'''
Todo: 
- Add the time to the NonRes.. Module
'''
class Residual(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, downsample=None):
        super(Residual, self).__init__()
        self.norm1 = nn.GroupNorm(min(32, input_dim), input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(32, output_dim), output_dim)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        output = self.relu(self.norm1(x))
        if self.downsample is not None:
            x = self.downsample(output)
        output = self.conv1(output)
        output = self.norm2(output)
        output = self.relu(output)
        output = self.conv2(output)

        return output + x


class NonResidual(nn.Module):
    ''' This will be our f.
        Same as Javiers Residual, but without being residual - that's done by the ode solver'''
    def __init__(self, input_dim, output_dim, stride=1):
        super(NonResidual, self).__init__()
        self.norm1 = nn.GroupNorm(min(32, input_dim), input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_dim+1, output_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(32, output_dim), output_dim)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False)

    def forward(self, t, x):
        shape = x.shape[-2:]
        t_ = torch.ones([1,1,*shape], dtype=torch.float32, device=x.device) * t
        #if len(x.shape) == 1:
        #    x = torch.reshape(x, [1, x.shape[0]])
        output = self.relu(self.norm1(x))
        output = torch.cat([output, t_], dim=1)
        output = self.conv1(output)
        output = self.norm2(output)
        output = self.relu(output)
        output = self.conv2(output)

        return output

class NonResidualNumpyCompat(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, device=None):
        super(NonResidualNumpyCompat, self).__init__()
        self.non_res_block = NonResidual(input_dim, output_dim, stride)
        self.device = device
        self.n_eval = 0 # for debugging. Edit: Not needed; the points used by the solver are also returned.

    def parameters(self, *args, **kwargs):
        return self.non_res_block.parameters(*args, **kwargs)

    def forward(self, t, x, shape=[64,7,7], batch_dim=1):
        '''Takes numpy arrays as input and gives np arrays as output'''
        x = torch.Tensor(x).to(self.device)
        t = torch.scalar_tensor(t).to(self.device)
        x = torch.reshape(x, [batch_dim, *shape])
        with torch.no_grad():
            result = self.non_res_block(t, x)
        return result.detach().cpu().numpy().reshape(-1)

class ODENetCore(autograd.Function):

    @staticmethod
    def forward(ctx, input, f, shape_params, rtol=1e-7, atol=1e-9, *f_parameters):
        """s
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # Todo: Need to define this externally
        # if not hasattr(ctx, "f"):
        #     self.f = NonResidual(input_dim=in_features + 1, output_dim=out_features)
        # t = torch.arange(6)
        t = [0, 5]
        # output = odeint(f, input.cpu(), t=t, rtol=rtol, atol=atol, tfirst=True) # todo: atol, hmax, hmin: what they mean, and then use them to reduce the required accuracy
        output = solve_ivp(f, t_span=t, y0=input.cpu(), method="LSODA", rtol=rtol, atol=atol, vectorized=True, min_step=0, ) # todo: atol, hmax, hmin: what they mean, and then use them to reduce the required accuracy
        ctx.f = f
        ctx.shape_params = shape_params # # list of shapes of all parameters of f
        ctx.output = output
        ctx.t = t
        ctx.rtol = rtol
        ctx.atol = atol
        import pdb
        pdb.set_trace()
        assert output.success, output.message + f"{output}"
        return output.y[:, -1]

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        import pdb
        pdb.set_trace()
        with torch.no_grad():
            output = ctx.output
            t = ctx.t
        s0 = [output[-1], grad_output, *[torch.zeros(*ps) for ps in ctx.shape_params]]
        def augmented_dynamics(cur_state, t):
            z_t = cur_state[0]
            a_t = cur_state[1]
            z = z_t.detach()
            z.requires_grad_(True)
            grad_f = torch.autograd.grad(ctx.f(z_t), [z, *ctx.f.parameters()] , allow_unused=True, retain_graph=True ) #  retain_graph=True : They use that in their implementation, but not sure what it does. allow_unused=True: If we need that, we should get an error when setting it to False. Can try with False.
            df_dz = grad_f[0]
            df_dtheta = grad_f[1:]
            return [ctx.f(z_t, t), -torch.einsum("i,i->", a_t, df_dz),
                    *[-torch.einsum("i,i->", a_t, df_dth_) for df_dth_ in df_dtheta] ]

        solution = solve_ivp(augmented_dynamics, t, s0)

        return solution[1], None, None, None, None, solution[2]  # Grad df/dtheta is solution[2]


class ODENetManual(nn.Module):
    def __init__(self, device, rtol=1e-7, atol=1e-9):
        super(ODENetManual, self).__init__()
        self.device = device
        self.rtol = rtol
        self.atol = atol
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.residual1 = Residual(64, 64, 2, nn.Conv2d(64, 64, kernel_size=1, stride=2, bias=False))
        self.residual2 = Residual(64, 64, 2, nn.Conv2d(64, 64, kernel_size=1, stride=2, bias=False))

        self.core = ODENetCore()

        self.norm1 = nn.GroupNorm(min(32, 64), 64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, 10)

        # self.f_torch = NonResidual(input_dim=64, output_dim=64) # ignoring t because the paper's code does that , too - EDIT: Todo, no I think it uses it!
        self.f_numpy = NonResidualNumpyCompat(input_dim=64, output_dim=64, device=device)
    # def f_numpy(self, x, t, shape=[64,7,7], batch_dim=1):
    #     '''Takes numpy arrays as input and gives np arrays as output'''
    #     x = torch.Tensor(x).to(self.device)
    #     t = torch.scalar_tensor(t).to(self.device)
    #     x = torch.reshape(x, [batch_dim, *shape])
    #     with torch.no_grad():
    #         result = self.f_torch(x, t)
    #     return result.detach().cpu().numpy().reshape(-1)
    def forward(self, x):
        device = x.device
        out = self.residual2(self.residual1(self.conv1(x)))
        shape = out.shape
        out = torch.reshape(out, [-1]) # [out.shape[0], -1]) # TODO: What to do with the batchsize? Set batchsize = 1? Can we vectorize the ODE-solving o
        # import pdb
        # pdb.set_trace()
        out_ = self.core.apply(out, self.f_numpy, [p.shape for p in self.f_numpy.parameters()], self.rtol, self.atol, *self.f_numpy.parameters())  #input, f, shape_params, rtol=1e-7, atol=1e-9
        #out = out_.y[:, -1] #np.reshape(out.y, [6, -1]) # there are six timesteps, I think this returns all of them
        out = torch.tensor(out, device=device, requires_grad=True, dtype=torch.float32)
        out = torch.reshape(out, shape)
        out = self.relu(self.norm1(out))
        out = self.pool(out)
        out = self.fc(torch.flatten(out, 1))
        return out