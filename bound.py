#%%
import torch
from NAQS import NAQS, intial_state
from lnZ_2d import TensorSampling, HonNN
import numpy as np
from scipy.linalg import sqrtm
from scipy import sparse
import csv
import time

num_site = 4
net_depth = 1
batch_size = 1000
l_r = 0.001
path = './data/' + 'betac_trial1/' + 'n%d_d%d_b%d_lr%f/' % (num_site, net_depth, batch_size, l_r)
model = torch.load(path + 'init_' + 'model_15000.pth')

#%%
def learning(model, model0, optimizer, loss_func, num_epoch, batch_size, my_device, beta=1.0, memo='init', save_path='./data/', B_anneal = 0.999):
    accuracy = 1
    model = model.to(my_device)
    model_optim = model
    # if model.n == model0.n:
    #     if len(model.layers) == len(model0.layers):
    #         model.load_state_dict(model0.state_dict())
    #     else:
    #         model.layers[0].weight.data = model0.layers[0].weight.detach().clone()
    #         model.layers[0].bias.data = model0.layers[0].bias.detach().clone()

    t0 = time.time()
    beta0 = beta
    for epoch in range(1, num_epoch + 1):

        beta = beta0 * (1 - B_anneal ** (epoch - 1))
        optimizer.zero_grad()
        with torch.no_grad():
            samples = model.sample(batch_size)
            # print(samples.shape)
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
                acc = abs(loss0 + 1)
                if acc < accuracy:
                    accuracy = acc
                    model_optim = model
            with open(save_path + memo +'.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, loss0,
                                 ])
            torch.save(model, save_path + memo + '_model_' + str(epoch) + '.pth')
            print('epoch= {}, checkpoint_loss= {}, time= {}'.format(epoch, loss0, t1-t0))

    return model_optim



#%%
def calc_H(num_site, site_index, mode='B', beta=1.0, my_type=torch.float64, my_device=torch.device('cuda:1')):
    B = np.array([[np.exp(beta), np.exp(-beta)], [np.exp(-beta), np.exp(beta)]])
    BHalf = sqrtm(B)
    I2 = np.eye(2)

    B = sparse.coo_matrix(B)
    BHalf = sparse.coo_matrix(BHalf)
    I2 = sparse.coo_matrix(I2)

    if mode == 'B':
        H = B.copy()
        for i in range(site_index):
            H = sparse.kron(I2, H, format='coo')
        for i in range(site_index + 1, num_site):
            H = sparse.kron(H, I2, format='coo')

        return H


    if mode == 'Bs':
        H = B.copy()
        for i in range(site_index):
            H = sparse.kron(I2, H, format='coo')
            # H = torch.kron(I2, H)

        for i in range(site_index + 1, num_site - site_index - 1):
            H = sparse.kron(H, I2, format='coo')
            # H = torch.kron(H, I2)
        # H = torch.kron(H, B)
        H = sparse.kron(H, B, format='coo')
        for i in range(num_site - site_index, num_site):
            H = sparse.kron(H, I2, format='coo')
        # H = torch.kron(H, I2)
        return H

    if mode == 'edge':
        B0 = np.array([[np.exp(beta), np.exp(-beta)], [np.exp(-beta), np.exp(beta)]])
        H = sparse.diags(B0.reshape(-1).repeat(2 ** (num_site - 2)), format='coo')

        return H
    if mode == '_B_':

        I = sparse.eye(2 ** (num_site - 2), format='coo')
        b1 = np.array([np.exp(beta), np.exp(-beta)])
        I1 = sparse.kron(I, b1, format='coo')
        b2 = np.array([np.exp(-beta), np.exp(beta)])
        I2 = sparse.kron(I, b2, format='coo')

        H = sparse.hstack([I1, I2], format='coo')


        return H



    if mode == 'bound':
        H = B.copy()
        for i in range(num_site):
            H = sparse.kron(H, I2, format='coo')
        h11 = ((np.array([[-1,1]]).repeat(2 ** site_index, axis=0)).repeat(2 ** (num_site - site_index), axis=1)).reshape(-1)
        h00 = np.array([-1, 1]).repeat(2 ** num_site)
        mask = sparse.coo_matrix((h00 * h11 + 1) / 2)
        H = ((H.T).multiply(mask)).T

        return H

    if mode == 'BHalf':
        H = BHalf.copy()
        for i in range(site_index):
            H = sparse.kron(I2, H, format='coo')
        for i in range(site_index + 1, num_site):
            H = sparse.kron(I2, H, format='coo')
        return H

    if mode == 'BHalfs':

        H = BHalf.copy()
        for i in range(site_index):
            H = sparse.kron(I2, H, format='coo')
            # H = torch.kron(I2, H)

        for i in range(site_index + 1, num_site - site_index - 1):
            H = sparse.kron(H, I2, format='coo')
            # H = torch.kron(H, I2)
        # H = torch.kron(H, B)
        H = sparse.kron(H, BHalf, format='coo')
        for i in range(num_site - site_index, num_site):
            H = sparse.kron(H, I2, format='coo')
        # H = torch.kron(H, I2)
        return H

    pass


#%%
# def eat_state_(num_site, net_depth, beta, batch_size, num_epoch, l_r, model0, my_type, my_device,save_path, memo='B_'):
#     log_norm = 0
#     model0 = model0.to(my_device)
#
#     HN = HonNN(num_site, beta=beta, my_type=my_type, my_device=my_device)
#     for site_index in range(num_site // 2):
#         site_indexes = [site_index, num_site - site_index - 1]
#         model = NAQS(num_site=num_site, net_depth=net_depth, my_type=my_type, my_device=my_device)
#
#         optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
#
#         H = lambda beta : calc_H(num_site, site_index, mode='Bs', beta=beta, my_type=my_type, my_device=my_device)
#         norm = lambda beta: HN.calc_norm(H(beta), model0)
#         log_norm += torch.log(norm(beta=beta))
#         loss_func = lambda samples, beta: - HN.B_on_sites(site_indexes, model0.psi_total, samples,
#                                                     beta=beta) / norm(beta=beta) / model.psi_total(samples)
#
#         model0 = learning(model, model0, optimizer, loss_func, num_epoch=num_epoch, batch_size=batch_size,
#                           my_device=my_device, beta=beta, memo=memo + '_' + str(site_index), save_path=save_path)
#
#         torch.save(model0, save_path + memo + '_model_' + str(site_index) + '.pth')
#         torch.save(torch.log(norm(beta=beta)), save_path + memo + '_log_norm_' + str(site_index) + '.pth')
#
#     return model0, log_norm

#%%
def _B_state(num_site, net_depth, beta, batch_size, num_epoch, l_r, model0, my_type, my_device, save_path, memo='_B_', B_anneal=0.999):
    log_norm = 0
    model0 = model0.to(my_device)
    HN = HonNN(num_site, beta=beta, my_type=my_type, my_device=my_device)

    # bound
    H = lambda beta: calc_H(num_site + 2, 0, mode='_B_', beta=beta, my_type=my_type, my_device=my_device)
    norm = lambda beta: HN.calc_norm(H(beta), model0)
    log_norm += torch.log(norm(beta=beta))
    model = NAQS(num_site=num_site, net_depth=net_depth, my_type=my_type, my_device=my_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
    loss_func = lambda samples, beta: - HN.B_on_middle(model0.psi_total, samples, beta=beta) / norm(
        beta=beta) / model.psi_total(samples)
    model0 = learning(model, model0, optimizer, loss_func, beta=beta, num_epoch=num_epoch, batch_size=batch_size,
                      my_device=my_device, memo=memo + '_' + str(0), save_path=save_path,
                      B_anneal=B_anneal)

    torch.save(model0, save_path + memo + '_model_' + str(0) + '.pth')
    torch.save(torch.log(norm(beta)), save_path + memo + '_log_norm_' + str(0) + '.pth')





    return model0, log_norm

def test():
    batch_size = 1000
    l_r = 0.001
    beta_c = 0.44068679350977147
    beta = beta_c

    num_epoch = 20000
    my_type = torch.float64
    my_device = torch.device('cuda:4')
    num_site = 4
    B_anneal = 0.9985

    path = './data/' + 'betac_trial1/' + 'n%d_d%d_b%d_lr%f/' % (num_site, 1, 1000, 0.001)
    data_saving_path = './data/' + 'ana_trial1/' + 'n%d_d%d_b%d_lr%f_a%f/' % (
    num_site, 3, batch_size, l_r, B_anneal)
    model0 = torch.load(data_saving_path + 'edge_' + 'model_0.pth')
    model0 = model0.to(my_device)
    HN = HonNN(num_site, beta=beta, my_type=my_type, my_device=my_device)
    TS = TensorSampling(num_site, beta=beta, my_type=my_type, my_device=my_device)

    samples = TS.all_sample(num_site)
    phi = HN.B_on_middle(model0.psi_total, samples, beta=beta)
    H = calc_H(num_site=num_site + 2, site_index=0, beta=beta, my_device=my_device, my_type=my_type, mode='_B_')
    print(H.todense())
    norm = HN.calc_norm(H, model0)
    print(norm)
    print(torch.sqrt(torch.sum(phi * phi)))


def main():
    import os
    batch_size = 1000
    l_r = 0.001
    beta_c = 0.44068679350977147
    beta = beta_c
    B_anneal = 0.9985

    num_epoch = 30000
    my_type = torch.float64
    my_device = torch.device('cuda:4')

    for num_site in [4]:
        for net_depth in [3]:
            path = './data/' + 'ana_trial1/' + 'n%d_d%d_b%d_lr%f_a%F/' % (num_site, 3, 1000, 0.001, 0.9985)
            model0 = torch.load(path + 'edge_' + 'model_%d.pth' % (0))

            data_saving_path = './data/' + 'ana_trial1/' + 'n%d_d%d_b%d_lr%f_a%f/' % (
            num_site, net_depth, batch_size, l_r, B_anneal)

            if not os.path.exists(data_saving_path):
                # os.mkdir(data_saving_path)
                os.makedirs(data_saving_path)

            _B_state(num_site, net_depth, beta, batch_size, num_epoch, l_r, model0, my_type, my_device,
                       save_path=data_saving_path, B_anneal=B_anneal)

if __name__ == '__main__':
    # test()
    main()