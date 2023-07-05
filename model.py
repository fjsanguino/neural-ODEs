import torch.nn as nn
import torch
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
    elif name == 'ResNet':
        return Residual()
    elif name == 'Paper':
        return PaperModel()


    else:
        print('No model with specified name')



    
