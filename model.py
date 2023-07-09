import torch.nn as nn
import torch

#import scipy.integrate
import sys

# import autograd.numpy as np
# from autograd.scipy.integrate import odeint
# from autograd import jacobian
# from autograd.builtins import tuple
# from torchdiffeq import odeint as odeint

#from autograd_utils import odeint

# import autograd.numpy as np
# from autograd.extend import primitive
# solve_ivp = primitive(scipy.integrate.solve_ivp)
# odeint = primitive(scipy.integrate.odeint)

from torch.autograd import Function
from scipy.integrate import solve_ivp

class PaperModel(nn.Module):

    def __init__(self):
        super(PaperModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.residual1 = Residual(64, 64, 2, nn.Conv2d(64, 64, kernel_size=1, stride=2, bias=False))
        self.residual2 = Residual(64, 64, 2, nn.Conv2d(64, 64, kernel_size=1, stride=2, bias=False))

        self.core = nn.Sequential(*[Residual(64, 64) for _ in range(6)])

        self.norm1 = nn.GroupNorm(min(32, 64), 64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        out = self.residual2(self.residual1(self.conv1(x)))
        out = self.core(out)
        out = self.relu(self.norm1(out))
        out = self.pool(out)
        out = self.fc(torch.flatten(out, 1))

        return out



class ODEIntFunction(Function):
    @staticmethod
    def forward(function, input, timeseries, method):
        out = solve_ivp(function, timeseries, input.flatten(), method)
        out = out.y[:,-1].reshape(32, 64, 7, 7)
        return torch.tensor(out, dtype=torch.float)

    @staticmethod
    def setup_context(ctx, inputs, output):
        function, input, timeseries, method = inputs
        ctx.save_for_backward(input, timeseries, output)
        ctx.function = function
        ctx.method = method
    
    @staticmethod
    def backward(ctx, grad_output):
        input, timeseries, output = ctx.saved_tensors
        #grad_input = torch.autograd.grad(ctx.function(timeseries, input), input)

        nz = ctx.function.parameters().shape
        
        aug_state = [torch.zeros(nz), output, grad_output]

        def augmented_dynamics(t, aug_output):
            z_t = aug_output[nz:nz+output.size]
            a_t = cur_state[nz+output.size:nz+output.size+grad_output.size]

            grad_func = torch.autograd.grad(ctx.function(t, z_t), z_t)

            return grad_func

        for i in range(len(timeseries) - 1, 0, -1):

            aug_state = scipy.integrate.solve_ivp(augmented_dynamics, aug_state, t[i-1:i+1].flip(0), method=ctx.method)

            aug_state[nz:nz+output.size] = output[i-1]
            aug_state[nz+output.size:nz+output.size+grad_output.size] += grad_output[i-1]
        
        return aug_state[nz:]

def odeint(function, input, timeseries, method='RK45'):
    return ODEIntFunction.apply(function, input, timeseries, method)



class ResidualODE(nn.Module):
    '''Residual layer that uses an ODE for constant-memory backpropagation'''
    def __init__(self, input_dim, output_dim, timestep=1, method='LSODA'):
        super(ResidualODE, self).__init__()
        self.timestep = torch.tensor([0, timestep], dtype=torch.float)
        self.method = method
        
        self.norm1 = nn.GroupNorm(input_dim, input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(output_dim, output_dim)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False)

    def odefunc(self, t, x):
        numpy = False
        if x.shape == (100352,):
            x = x.reshape((32, 64, 7, 7))
            x = torch.tensor(x, dtype=torch.float)
            numpy = True
            
        output = self.relu(self.norm1(x))

        output = self.conv1(output)
        output = self.norm2(output)
        output = self.relu(output)
        output = self.conv2(output)

        out = output + x

        if numpy:
            return out.flatten().numpy()
        else:
            return out
        
    def forward(self, x):
        return odeint(self.odefunc, x, self.timestep, method=self.method)
    
    
    
# class ODEFunc(nn.Module):
#     def __init__(self, odelayer):
#         super().__init__()
#         self.odelayer = odelayer

#     def forward(self, t, x):
#         return self.odelayer(x)
        
    
class ODEnet(nn.Module):

    def __init__(self):
        super(ODEnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.residual1 = Residual(64, 64, 2, nn.Conv2d(64, 64, kernel_size=1, stride=2, bias=False))
        self.residual2 = Residual(64, 64, 2, nn.Conv2d(64, 64, kernel_size=1, stride=2, bias=False))

        #self.core = nn.Sequential(*[Residual(64, 64) for _ in range(6)])
        # self.odelayer = Residual(64, 64)

        # self.odefunc = ODEFunc(self.odelayer)
        self.odelayer = ResidualODE(64, 64, timestep=6, method='LSODA')

        self.norm1 = nn.GroupNorm(min(32, 64), 64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, 10)
        
    # def odefunc (self, t, x):
    #     return self.odelayer(x)

    def forward(self, x):
        out = self.residual2(self.residual1(self.conv1(x)))

        #out = self.core(out)
        #out = self.ODEsolve(out, self.odefunc, 0, 6)
        out = self.odelayer(out)
        
        out = self.relu(self.norm1(out))
        out = self.pool(out)
        out = self.fc(torch.flatten(out, 1))

        return out

    # def ODEsolve(self, inp, func, start, end):
    #     #return solve_ivp(func, (start, end), inp, 'LSODA')

    #     timeseries = torch.tensor([start, end], dtype=torch.float64).to(inp.device)
        
    #     #return odeint(func, inp, timeseries, method='implicit_adams')[1]
    #     return odeint(func, inp, timeseries, method='implicit_adams')

    #     # r = ode(func).set_integrator('vode', method='adams')
    #     # r.set_initial_value(inp, start)

    #     # return r.integrate(end)


class Residual(nn.Module):
    expansion = 1

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

class MLP(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_dim,300)
        self.linear2 = nn.Linear(300,output_dim)

    def forward(self,x):
        out = self.flatten(x)
        out = self.linear1(out)
        out = self.linear2(out)

        return out

def get_model(name, input_dim = 28*28, output_dim = 10):
    if name == 'MLP':
        return MLP(input_dim,output_dim)
    elif name == 'Paper':
        return PaperModel()
    elif name == 'ODEnet':
        return ODEnet()


    else:
        print('No model with specified name \"' ,name , '\" exiting code...')
        sys.exit()
