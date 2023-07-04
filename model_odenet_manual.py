import torch
from torch import nn, autograd
import math
import numpy as np
from scipy.integrate import odeint, solve_ivp



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
    def __init__(self, input_dim, output_dim, stride=1, downsample=None):
        super(NonResidual, self).__init__()
        self.norm1 = nn.GroupNorm(min(32, input_dim), input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(32, output_dim), output_dim)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False)

    def forward(self, x, t=None):
        if len(x.shape) == 1:
            x = torch.reshape(x, [1, x.shape[0]])
        output = self.relu(self.norm1(x))
        if self.downsample is not None:
            output = self.downsample(output)
        output = self.conv1(output)
        output = self.norm2(output)
        output = self.relu(output)
        output = self.conv2(output)

        return output

class ODENetCore(autograd.Function):

    @staticmethod
    def forward(ctx, input, f, shape_params):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # Todo: Need to define this externally
        # if not hasattr(ctx, "f"):
        #     self.f = NonResidual(input_dim=in_features + 1, output_dim=out_features)
        output = odeint(f, input.cpu().detach(), t=np.arange(6) )

        ctx.f = f
        ctx.shape_params = shape_params # # list of shapes of all parameters of f
        ctx.save_for_backward(output, t=np.arange(6))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        with torch.no_grad():
            output, t = ctx.saved_tensors

        s0 = [output[-1], grad_output, *[torch.zeros(*ps) for ps in ctx.shape_params]]
        def augmented_dynamics(cur_state, t):
            z_t = cur_state[0]
            a_t = cur_state[1]
            z = z_t.detach()
            z.requires_grad_(True)
            grad_f = torch.autograd.grad(ctx.f(z_t), [z, *ctx.f.parameters()] )
            df_dz = grad_f[0]
            df_dtheta = grad_f[1:]
            return [ctx.f(z_t, t), -torch.einsum("i,i->", a_t, df_dz),
                    *[-torch.einsum("i,i->", a_t, df_dth_) for df_dth_ in df_dtheta] ]

        solution = solve_ivp(augmented_dynamics, t, s0)

        return solution[1], solution[2], None


class ODENetManual(nn.Module):
    def __init__(self):
        super(ODENetManual, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.residual1 = Residual(64, 64, 2, nn.Conv2d(64, 64, kernel_size=1, stride=2, bias=False))
        self.residual2 = Residual(64, 64, 2, nn.Conv2d(64, 64, kernel_size=1, stride=2, bias=False))

        self.core = ODENetCore()

        self.norm1 = nn.GroupNorm(min(32, 64), 64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, 10)

        self.f = NonResidual(input_dim=64, output_dim=64) # ignoring t because the paper's code does that , too

    def forward(self, x):

        out = self.residual2(self.residual1(self.conv1(x)))
        out = torch.reshape(out, [-1]) # [out.shape[0], -1]) # TODO: What to do with the batchsize? Set batchsize = 1? Can we vectorize the ODE-solving o        out = self.core.apply(out, self.f, [p.shape for p in self.f.parameters()])
        out = self.core.apply(out, self.f, [p.shape for p in self.f.parameters()])
        out = torch.reshape(out, [1, -1])
        out = self.relu(self.norm1(out))
        out = self.pool(out)
        out = self.fc(torch.flatten(out, 1))
        return out