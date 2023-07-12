import torch
from torch import nn, autograd
import math
import numpy as np
from scipy.integrate import odeint, solve_ivp
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)

# Try using faster paper solver
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
        #if len(x.shape) == 1:
        #    x = torch.reshape(x, [1, x.shape[0]])
        output = self.relu(self.norm1(x))
        output = torch.cat([output, t_], dim=1)
        output = self.conv1(output)
        output = self.norm2(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.norm3(output) # added
        return output

class NonResidualNumpyCompat(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, device=None, conv=True, shape=[64,7,7]):
        super(NonResidualNumpyCompat, self).__init__()
        assert input_dim == output_dim
        if conv:
            self.non_res_block = NonResidual(input_dim, output_dim, stride)
        else:
            self.non_res_block = NonResidualFC(input_dim, output_dim)

        self.shape = shape
        self.device = device
        self.n_eval = 0 # for debugging. Edit: Not needed; the points used by the solver are also returned.

   # def parameters(self, *args, **kwargs):
   #     return self.non_res_block.parameters(*args, **kwargs)

    def forward(self, t, x, batch_dim=1):
        '''Takes numpy arrays as input and gives np arrays as output'''
        x = torch.Tensor(x).to(self.device)
        t = torch.scalar_tensor(t).to(self.device)
        x = torch.reshape(x, [batch_dim, *self.shape])
        with torch.no_grad():
            result = self.non_res_block(t, x)
        return result.detach().cpu().numpy().reshape(-1)

class ODENetCore(autograd.Function):

    @staticmethod
    def forward(ctx, input, f, shape_params, input_shape, rtol=1e-2, atol=1e-1, *f_parameters):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # Todo: Need to define this externally
        # if not hasattr(ctx, "f"):
        #     self.f = NonResidual(input_dim=in_features + 1, output_dim=out_features)
        # t = torch.arange(6)
        #t = [0, 5]
        t = torch.tensor([0, 5], dtype=torch.float)
        # output = odeint(f, input.cpu(), t=t, rtol=rtol, atol=atol, tfirst=True) # todo: atol, hmax, hmin: what they mean, and then use them to reduce the required accuracy
        with torch.no_grad():
            #output = solve_ivp(f, t_span=t, y0=input.cpu().numpy(), method="LSODA", rtol=rtol, atol=atol, vectorized=True, min_step=0, ) # todo: atol, hmax, hmin: what they mean, and then use them to reduce the required accuracy
            output = tdeq._impl.fixed_adams.AdamsBashforthMoulton(f, y0=input, perturb=False, rtol=rtol, atol=atol).integrate(t)
        ctx.f = f
        ctx.shape_params = shape_params # # list of shapes of all parameters of f
        ctx.t = t
        ctx.rtol = rtol
        ctx.atol = atol
        ctx.input_shape = input_shape
        #assert output.success, output.message + f"{output}"
        #output =  torch.tensor(output.y[:, -1], dtype=torch.float32, device=input.device, requires_grad=True)
        #output =  torch.tensor(output[:, -1], dtype=torch.float32, device=input.device, requires_grad=True)
        output = output[:, -1]
        
        ctx.output = output
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """        
        with torch.no_grad():
            output = ctx.output
            t = ctx.t
            s0 = [output.detach().cpu().numpy(), grad_output.detach().cpu().numpy(), *[np.zeros(ps) for ps in ctx.shape_params]]
            s0_shapes = [s.shape for s in s0]
            s0_sizes = [np.prod(s) for s in s0_shapes]
            #s0 = [el.flatten() for el in s0]
            #s0 = np.concatenate(s0)
            def augmented_dynamics(t, cur_state, device):
                #z_t = cur_state[:s0_sizes[0]]
                z_t = cur_state[0]
                #a_t = cur_state[ s0_sizes[0] : s0_sizes[0]+s0_sizes[1] ]
                a_t = cur_state[1]
                #z_t = torch.tensor(z_t, dtype=torch.float32, device=device)
                #a_t = torch.tensor(a_t, dtype=torch.float32, device=device)
                t = torch.scalar_tensor(t, requires_grad=False, dtype=torch.float32, device=device)
                # if len(s0_shapes[0])>1:
                #     z_t = z_t.reshape(s0_shapes[0])
                # if len(s0_shapes[1])>1:
                #     a_t = a_t.reshape(s0_shapes[1])
                with torch.enable_grad():
                    #z = z_t.detach()[None, ...]
                    z.requires_grad_(True)
                    grad_f = []
                    #f_applied = ctx.f.non_res_block(t, torch.reshape(z, ctx.input_shape)) # ctx.f.non_res_block(t, z)
                    f_applied = ctx.f(t, z)
                    # TODO: Use f.non_res_block and reshape z correctly before inputting it here.
                    # TODO: Cannot use f, because we need tensors for autograd and f expects numpy arrays.
                    for f_val in f_applied.flatten():
                        grad_f.append(torch.autograd.grad(f_val, [z, *ctx.f.parameters()] , allow_unused=True,
                                                          retain_graph=True )) #  retain_graph=True : They use that in their implementation, but not sure what it does. allow_unused=True: If we need that, we should get an error when setting it to False. Can try with False.
                df_dz = torch.stack([grad_f[i][0].flatten() for i in range(len(grad_f))], dim=0) # each row is for one element of f.
                # TODO: There is more than one dimension in grad_f[i]. Aha, it's a tuple of tensors. each can have any length.
                df_dtheta = []
                for j in range(1,len(grad_f[0])):
                    df_dtheta.append(torch.stack([grad_f[i][j].flatten() for i in range(len(grad_f))], dim=0) )# each row is for one element of f.
                # df_dtheta = grad_f[1:]
                return np.concatenate([ctx.f(t, z_t).flatten(), # unnecessary additional function call?
                                       -torch.einsum("i,ij->j", a_t, df_dz).detach().cpu().numpy().flatten(),
                        *[-torch.einsum("i,ij->j", a_t, df_dth_).detach().cpu().numpy().flatten() for df_dth_ in df_dtheta] ])
            #solution = solve_ivp(lambda t, s: augmented_dynamics(t, s, output.device), t, s0)
            solution = tdeq.AdamsBashforthMoulton(lambda t, s: augmented_dynamics(t, s, output.device), t, s0)
            solution = solution.y[:, -1]
            sol_elems = []
            start_idx = 0
            for i, (shp, sz) in enumerate(zip(s0_shapes, s0_sizes)):
                elem = solution[start_idx : start_idx + sz]
                start_idx = start_idx + sz
                elem = torch.tensor(elem, dtype=torch.float32, device=output.device, requires_grad=False)
                if len(shp) > 1:
                    elem = elem.reshape(shp)
                sol_elems.append(-elem) # by trial and error, this has to be negative for the loss to decrease.
        # TODO: It is unclear whether the first element, sol_elems[1], has to be negative as well.
        return sol_elems[1], None, None, None, None, None, *sol_elems[2:],   # Grad df/dtheta is solution[2]


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
        self.f_torch = NonResidual(input_dim=64, output_dim=64)
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
        #out = torch.reshape(out, [-1]) # [out.shape[0], -1]) # TODO: What to do with the batchsize? Set batchsize = 1? Can we vectorize the ODE-solving o
        # import pdb
        # pdb.set_trace()
        #out_ = self.core.apply(out, self.f_numpy, [p.shape for p in self.f_numpy.parameters()], shape, self.rtol, self.atol, *self.f_numpy.parameters())  #input, f, shape_params, rtol=1e-7, atol=1e-9
        out_ = self.core.apply(out, self.f_torch, [p.shape for p in self.f_numpy.parameters()], shape, self.rtol, self.atol, *self.f_torch.parameters())
        #out = out_.y[:, -1] #np.reshape(out.y, [6, -1]) # there are six timesteps, I think this returns all of them
        #out = torch.tensor(out_, device=device, requires_grad=True, dtype=torch.float32)
        #out = torch.reshape(out_, shape)
        out = self.relu(self.norm1(out))
        out = self.pool(out)
        out = self.fc(torch.flatten(out, 1))
        return out
