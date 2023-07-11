import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    batch_size = 32
    increase_batch_size_every = 1000#20
    decrease_lr_every = 2

    f = NonResidualNumpyCompat(input_dim=2, output_dim=2, shape=[2,], conv=False)
    core = ODENetCore()

    optimizer = torch.optim.SGD(f.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(f.parameters(), lr=0.001)
    # optimizer = torch.optim.RMSprop(f.parameters(), lr=0.001)
    train_data = TensorDataset(torch.tensor(Z0), torch.tensor(Y))
    eval_data = TensorDataset(torch.tensor(Z0_eval), torch.tensor(Y_eval))

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=True)
    val_loader = DataLoader(eval_data, batch_size=1, shuffle=True, drop_last=True)
    for epoch in range(epochs):
        if (epoch + 1) % increase_batch_size_every == 0:
            batch_size *= 2
        if (epoch + 1) % decrease_lr_every == 0:
            optimizer.param_groups[0]["lr"] *= 0.1
            print(f"Ep {epoch}; lr: {optimizer.param_groups[0]['lr']}")
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
                # for param in f.parameters():
                #     g = param.grad
                #     print(g)
                try:
                    print(f"Loss: {loss.cpu() / l_ct}")
                except ZeroDivisionError:
                    print(f"Ep. {epoch}, index {idx}: loss-count was zero for some reason. Total loss sum: {loss.cpu()}; loss count: {l_ct}. Batchsize: {batch_size}")
                optimizer.zero_grad()
                loss = 0.
                l_ct = 0
                # if len(list(f.parameters())) not in [1,2]:
                #     raise AssertionError("with simple=True the core network should have only one linear layer (plus bias maybe)")
                # A_learned = list(f.parameters())[0]
                # try:
                #     b_learned = list(f.parameters())[1]
                # except IndexError:
                #     pass
                # bla = torch.svd(A_learned)
                # if epoch >= 1 and (idx  == 0):
                #     import pdb
                #     pdb.set_trace()
        loss = loss if type(loss) == float else loss.cpu()
        try:
            print(f"\nEp {epoch}: Loss: {loss/ l_ct}\n")
        except ZeroDivisionError:
            print(
                f"Ep. {epoch}, index {idx}: loss-count was zero for some reason. Total loss sum: {loss}; loss count: {l_ct}. Batchsize: {batch_size}")

        # Check for proximity to true solution



if __name__ == '__main__':
    test_odenet_core()


