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
    def forward(ctx, input, f, rtol=1e-2, atol=1e-1):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        t = torch.tensor([0, 5], dtype=torch.float)
        with torch.no_grad():
            output = tdeq._impl.fixed_adams.AdamsBashforthMoulton(f, y0=input, perturb=False, rtol=rtol, atol=atol).integrate(t)[1]

        ctx.f = f
        ctx.rtol = rtol
        ctx.atol = atol

        ctx.save_for_backward(t, output, *f.parameters())

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

            #zeros = torch.cat([torch.zeros_like(param) for param in ctx.f.parameters()])
            s0 = [output, grad_output, *[torch.zeros_like(param).flatten() for param in params]]
            s0 = torch.cat([s0_.flatten() for s0_ in s0])

            # REALLY PYTORCH??? REALLY??
            os = output.flatten().shape[0]
            gos = grad_output.flatten().shape[0]

            def augmented_dynamics(t, cur_state, perturb=tdeq._impl.misc.Perturb.NONE):
                assert(perturb == tdeq._impl.misc.Perturb.NONE)

                z_t = cur_state[:os].reshape(output.shape)
                a_t = cur_state[os:os+gos].reshape(grad_output.shape)

                t = torch.scalar_tensor(t, requires_grad=False, dtype=torch.float32)

                with torch.enable_grad():
                    z_t = z_t.requires_grad_(True)
                    t = t.requires_grad_(True)

                    # for f_val in f_applied.flatten():
                    #     grad_f.append(torch.autograd.grad(f_val, [z_t, *ctx.f.parameters()] , allow_unused=True,
                    #                                       retain_graph=True )) #  retain_graph=True : They use that in their implementation, but not sure what it does. allow_unused=True: If we need that, we should get an error when setting it to False. Can try with False.

                    grad_z, *grad_params = torch.autograd.grad(ctx.f(t, z_t), [z_t, *params], -a_t, allow_unused=True, retain_graph=True)

                grad_f = [grad_z, a_t, *grad_params]
                    
                grad_f = torch.cat([grad_f_.flatten() for grad_f_ in grad_f])

                return grad_f

                #df_dz = torch.stack([grad_f[i][0].flatten() for i in range(len(grad_f))], dim=0) # each row is for one element of f.
                # df_dtheta = []
                # for j in range(1,len(grad_f[0])):
                #     df_dtheta.append(torch.stack([grad_f[i][j].flatten() for i in range(len(grad_f))], dim=0) )# each row is for one element of f.
                #df_dtheta = grad_f

                # return torch.cat((ctx.f(t, z_t), -torch.einsum("i,ij->j", a_t, df_dz), -torch.einsum("i,ij->j", a_t, df_dtheta)))

                # return np.concatenate([ctx.f(t, z_t).flatten(), # unnecessary additional function call?
                #                        -torch.einsum("i,ij->j", a_t, df_dz).detach().cpu().numpy().flatten(),
                #         *[-torch.einsum("i,ij->j", a_t, df_dth_).detach().cpu().numpy().flatten() for df_dth_ in df_dtheta] ])

            #solution = tdeq.AdamsBashforthMoulton(lambda t, s: augmented_dynamics(t, s, output.device), t, s0)
            solution = tdeq._impl.fixed_adams.AdamsBashforthMoulton(augmented_dynamics, y0=s0, perturb=False, rtol=ctx.rtol, atol=ctx.atol).integrate(t.flip(0))[1]

        #     sol_elems = []
        #     start_idx = 0
        #     for i, (shp, sz) in enumerate(zip(s0_shapes, s0_sizes)):
        #         elem = solution[start_idx : start_idx + sz]
        #         start_idx = start_idx + sz
        #         elem = torch.tensor(elem, dtype=torch.float32, device=output.device, requires_grad=False)
        #         if len(shp) > 1:
        #             elem = elem.reshape(shp)
        #         sol_elems.append(-elem) # by trial and error, this has to be negative for the loss to decrease.
        # # TODO: It is unclear whether the first element, sol_elems[1], has to be negative as well.
        # return sol_elems[1], None, None, None, None, None, *sol_elems[2:],   # Grad df/dtheta is solution[2]

        return solution[:os].reshape(output.shape), None, None, None


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

        #out = self.core.apply(out, self.f_torch, [p.shape for p in self.f_torch.parameters()], shape, self.rtol, self.atol, *self.f_torch.parameters())
        out = self.core.apply(out, self.f_torch, self.rtol, self.atol)

        out = self.relu(self.norm1(out))
        out = self.pool(out)
        out = self.fc(torch.flatten(out, 1))
        return out
