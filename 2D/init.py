import torch
from NAQS import NAQS, intial_state
from lnZ_2d import TensorSampling, HonNN
import numpy as np
from scipy.linalg import sqrtm
from scipy import sparse
import csv
import time
def learning(model, model0, optimizer, loss_func, num_epoch, batch_size, my_device, beta=1.0, memo='init', save_path='./data/', anneal=False):
    accuracy = 1
    model = model.to(my_device)
    model_optim = model
    if model0:
        if model.n == model0.n:
            if len(model.layers) == len(model0.layers):
                model.load_state_dict(model0.state_dict())
            else:
                model.layers[0].weight.data = model0.layers[0].weight.detach().clone()
                model.layers[0].bias.data = model0.layers[0].bias.detach().clone()


    t0 = time.time()
    beta0 = beta
    for epoch in range(1, num_epoch + 1):

        beta = beta0 * (1 - 0.999 ** (epoch - 1))
        optimizer.zero_grad()
        with torch.no_grad():
            samples = model.sample(batch_size)
        assert not samples.requires_grad

        with torch.no_grad():
            loss = loss_func(samples, beta)
        assert not loss.requires_grad
        log_psi = model.log_psi_conj(samples)
        loss_reinforce = - ((2 * (loss - loss.mean()) * log_psi).mean() / loss.mean()).real

        loss_reinforce.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            t1 = time.time()
            with torch.no_grad():
                samples0 = model.sample(batch_size * 100)
                loss0 = loss_func(samples0, beta)
                loss0 = loss0.mean().item()
                if abs(loss0 + 1) < accuracy:
                    accuracy = abs(loss0 + 1)
                    model_optim = model
            with open(save_path + memo +'.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, loss0,
                                 ])
            torch.save(model, save_path + memo + '_model_' + str(epoch) + '.pth')
            print('epoch= {}, checkpoint_loss= {}, time= {}'.format(epoch, loss0, t1-t0))

    return model_optim

def learn_init_state(num_site, net_depth, beta, batch_size, num_epoch, l_r, my_type, my_device, save_path, memo='init'):
    model = NAQS(num_site=num_site, net_depth=net_depth, my_type=my_type, my_device=my_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=l_r)

    TS = TensorSampling(num_site, beta, my_type, my_device)
    psi0, log_norm = TS.psi0()
    loss_func = lambda samples, beta: - TS.phi_func(psi0, samples) / model.psi_total(samples)

    model_out = learning(model=model, optimizer=optimizer, loss_func=loss_func, model0=None, beta=beta, num_epoch=num_epoch, batch_size=batch_size, my_device=my_device, memo=memo, save_path=save_path)

    torch.save(model_out, save_path + memo + '_model_' + '.pth')
    torch.save(log_norm, save_path + memo + '_log_norm_' + '.pth')

    return model_out, log_norm


def main():
    import os
    batch_size = 1000
    l_r = 0.001
    beta_c = 0.44068679350977147
    beta = beta_c
    net_depth = 1

    num_epoch = 20000
    my_type = torch.float64
    my_device = torch.device('cuda:4')

    for num_site in [4,6]:
        data_saving_path = './data/' + 'ana_trial0/' + 'n%d_d%d_b%d_lr%f/' % (num_site, net_depth, batch_size, l_r)

        if not os.path.exists(data_saving_path):
            os.makedirs(data_saving_path)

        learn_init_state(
            num_site=num_site,
            net_depth=3,
            beta=beta,
            batch_size=batch_size,
            num_epoch=num_epoch,
            l_r=l_r,
            my_type=my_type,
            my_device=my_device,
            save_path=data_saving_path,
            memo='init'

        )

if __name__ == '__main__':
    main()