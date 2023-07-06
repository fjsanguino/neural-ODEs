import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model_odenet_manual import ODENetCore, NonResidualNumpyCompat


class Simple2DODE():
    def __init__(self, d1, d2):
        ''' ODE dynamics: dz/dt = A*z, where A is diagonalizable and 2D.'''
        self.d1 = d1
        self.d2 = d2
        self.U = np.array([[1, -1],[1,1]]) * 1/np.sqrt(2)
        self.A = self.U @ np.diag([d1, d2]) @ self.U.T

    def simulate_true(self, Z0, T):
        ''' Z0: Matrix of K 2D initial conditions. Time is assumed to start at 0.
            T: end time.
        '''
        assert Z0.shape[1] == 2
        Z = Z0 * np.array([np.exp(self.d1 * T), np.exp(self.d2 * T)])
        return np.einsum("ij,kj->ki", self.U, Z)

    def random_data(self, K, T):
        Z0 = np.random.random((K, 2))*2. - 1. # -> values in [-1, 1]
        Z1 = self.simulate_true(Z0, T)
        return Z0, Z1


def test_odenet_core():
    ode = Simple2DODE(d1=0.5, d2=-1)
    Z0, Y = ode.random_data(K=1000, T=1.)
    Z0_eval, Y_eval = ode.random_data(K=200, T=1.)
    epochs = 100
    rtol= 1e-7
    atol=1e-9
    batch_size = 8

    f = NonResidualNumpyCompat(input_dim=2, output_dim=2, shape=[2,], conv=False)
    core = ODENetCore()

    optimizer = torch.optim.SGD(f.parameters(), lr=0.001, momentum=0.9)
    train_data = TensorDataset(torch.tensor(Z0), torch.tensor(Y))
    eval_data = TensorDataset(torch.tensor(Z0_eval), torch.tensor(Y_eval))

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=True)
    val_loader = DataLoader(eval_data, batch_size=1, shuffle=True, drop_last=True)
    for epoch in range(epochs):
        l_ct = 0
        l_sum = 0
        optimizer.zero_grad()
        loss = 0
        for idx, (x, y) in enumerate(train_loader):
            pred = core.apply(x.flatten(), f, [p.shape for p in f.parameters()], x.shape , rtol, atol, *f.parameters())
            loss += torch.mean(torch.abs(pred - y))
            l_ct += 1
            if l_ct >= batch_size:
                (loss / l_ct).backward()
                optimizer.step()
                for param in f.parameters():
                    g = param.grad
                    print(g)
                print(f"Loss: {loss.cpu() / l_ct}")
                optimizer.zero_grad()
                loss = 0.
                l_ct = 0
        print(f"Ep {epoch}: Loss: {l_sum/l_ct}")



if __name__ == '__main__':
    test_odenet_core()


