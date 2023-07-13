import torch
from torch import nn, autograd
import math
import numpy as np

# Use the paper ode solver because it is much faster and integrates better with torch
# Note that we ONLY use the ported ODE solver, we implement the adjoint and backprop ourselves.
import torchdiffeq as tdeq

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

class NonResidualFC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NonResidualFC, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_dim * 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, t, x):
        t_ = torch.ones(x.shape, dtype=torch.float32, device=x.device) * t
        out = self.fc1(torch.cat([x, t_], dim=1))
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


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
        self.norm3 = nn.GroupNorm(min(32, output_dim), output_dim) # paper has another norm in the end

    def forward(self, t, x, perturb=tdeq._impl.misc.Perturb.NONE):
        assert(perturb == tdeq._impl.misc.Perturb.NONE)

        shape = x.shape[-2:]
        t_ = torch.ones([x.shape[0],1,*shape], dtype=torch.float32, device=x.device) * t
        output = self.relu(self.norm1(x))
        output = torch.cat([output, t_], dim=1)
        output = self.conv1(output)
        output = self.norm2(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.norm3(output) # added
        return output

class ODENetCore(autograd.Function):
    @staticmethod
    def forward(ctx, input, t, f, rtol=1e-2, atol=1e-1, *params):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        with torch.no_grad():
            output = tdeq._impl.fixed_adams.AdamsBashforthMoulton(f, y0=input, perturb=False, rtol=rtol, atol=atol).integrate(t)[1]

        ctx.f = f
        ctx.rtol = rtol
        ctx.atol = atol

        ctx.save_for_backward(t, output, *params)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        with torch.no_grad():
            t, output, *params = ctx.saved_tensors
 
            s0 = [output,
                  grad_output,
                  *[torch.zeros_like(param).flatten() for param in params]]            
            s0 = torch.cat([s0_.flatten() for s0_ in s0])
            
            os = output.flatten().shape[0]
            gos = grad_output.flatten().shape[0]

            def augmented_dynamics(t, cur_state, perturb=tdeq._impl.misc.Perturb.NONE):
                assert(perturb == tdeq._impl.misc.Perturb.NONE)

                z_t = cur_state[:os].reshape(output.shape)
                a_t = cur_state[os:os+gos].reshape(grad_output.shape)

                with torch.enable_grad():
                    z_t = z_t.requires_grad_(True)
                    t = t.requires_grad_(True)

                    ftz = ctx.f(t, z_t)
                    
                    grad_z, *grad_params = torch.autograd.grad(ftz, [z_t, *params], -a_t, allow_unused=True, retain_graph=True)

                grad_f = [ftz, grad_z, *grad_params]
                    
                grad_f = torch.cat([grad_f_.flatten() for grad_f_ in grad_f])

                return grad_f

            solution = tdeq._impl.fixed_adams.AdamsBashforthMoulton(augmented_dynamics, y0=s0, perturb=False, rtol=ctx.rtol, atol=ctx.atol).integrate(t.flip(0))[1]

            dfdz = solution[os:os+gos].reshape(output.shape)
        
            dfdtheta = []
            start_idx = os+gos
            for param in params:
                size = param.flatten().shape[0]
                shape = param.shape
                stop_idx = start_idx + size
                
                dfdtheta.append(solution[start_idx:stop_idx].reshape(shape))
                start_idx = stop_idx
        
        return dfdz, None, None, None, None, *dfdtheta


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

        self.f_torch = NonResidual(input_dim=64, output_dim=64)

    def forward(self, x):
        device = x.device
        out = self.residual2(self.residual1(self.conv1(x)))

        t = torch.tensor([0, 5], dtype=torch.float)
        
        out = self.core.apply(out, t, self.f_torch, self.rtol, self.atol, *self.f_torch.parameters())

        out = self.relu(self.norm1(out))
        out = self.pool(out)
        out = self.fc(torch.flatten(out, 1))
        return out
