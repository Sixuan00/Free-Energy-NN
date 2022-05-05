#%%
import torch
from NAQS import NAQS, intial_state
from lnZ_2d import TensorSampling, HonNN
import numpy as np
from scipy.linalg import sqrtm
from scipy import sparse
import csv
import time


#%%
def learning(model, optimizer, loss_func, num_epoch, batch_size, my_device, beta=1.0, memo='init', save_path='./data/', anneal=False):
    model = model.to(my_device)
    t0 = time.time()
    for epoch in range(1, num_epoch + 1):

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
        # with torch.no_grad():
        #     optimizer.param_groups[0]['lr'] *= (1 - (1 / 100) ** (1 / num_epoch))


        # print('Epoch: {}, loss_reinforce: {}, loss:{}'.format(epoch, loss_reinforce,  loss.mean()))

        if epoch % 1000 == 0:
            t1 = time.time()
            with torch.no_grad():
                samples0 = model.sample(batch_size * 100)
                loss0 = loss_func(samples0, beta)
                loss0 = loss0.mean().item()
            # losses[int(epoch / 1000) - 1] = loss0
            with open(save_path + memo +'.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, loss0,
                                 ])
            print('epoch= {}, checkpoint_loss= {}, time= {}'.format(epoch, loss0, t1-t0))

    return model
#%%

def calc_H(num_site, site_index, mode='B', beta=1.0, my_type=torch.float64, my_device=torch.device('cuda:1')):
    # beta = torch.tensor([beta], dtype=my_type, device=my_device)
    # B = torch.tensor([[torch.exp(beta), torch.exp(-beta)], [torch.exp(-beta), torch.exp(beta)]], dtype=my_type,
    #                  device=my_device)
    # BHalf = torch.from_numpy(sqrtm(B.cpu().numpy())).type(my_type).to(my_device)
    B = np.array([[np.exp(beta), np.exp(-beta)], [np.exp(-beta), np.exp(beta)]])
    BHalf = sqrtm(B)
    I2 = np.eye(2)

    B = sparse.coo_matrix(B)
    BHalf = sparse.coo_matrix(BHalf)
    I2 = sparse.coo_matrix(I2)

    # I2 = torch.eye(2, dtype=my_type, device=my_device)
    if mode == 'B':
        # H = B.detach().clone()
        H = B.copy()
        for i in range(site_index):
            H = sparse.kron(I2, H, format='coo')
            # H = torch.kron(I2, H)

        for i in range(site_index + 1, num_site):
            H = sparse.kron(H, I2, format='coo')
            # H = torch.kron(H, I2)
        # H = torch.kron(H, B)
        # H = sparse.kron(H, B, format='coo')
        # for i in range(num_site - site_index, num_site):
        #     H = sparse.kron(H, I2, format='coo')
            # H = torch.kron(H, I2)

        return H

    if mode == 'edge':
        # I1 = torch.ones([4, 2 ** (num_site - 2)], dtype=my_type, device=my_device)
        # H = torch.diag(((I1.T * B.view(-1)).T).view(-1))
        # I1 = np.ones([4, 2 ** (num_site - 2)])
        # H = np.diag(((I1.T * B.view(-1)).T).view(-1))
        B0 = np.array([[np.exp(beta), np.exp(-beta)], [np.exp(-beta), np.exp(beta)]])
        H = sparse.diags(B0.reshape(-1).repeat(2 ** (num_site - 2)), format='coo')

        return H

    if mode == 'bound':
        # H = B.detach().clone()
        H = B.copy()
        for i in range(num_site):
            H = sparse.kron(H, I2, format='coo')
            # H = torch.kron(H, I2)
        # h0 = - torch.ones([2 ** site_index, 2 ** (num_site - site_index)], dtype=my_type, device=my_device)
        # h11 = torch.cat([h0, -h0], dim=1).view(-1)
        #
        # h0 = - torch.ones(2 ** num_site, dtype=my_type, device=my_device)
        # h00 = torch.cat([h0, -h0])
        #
        # mask = (h00 * h11 + 1) / 2
        #
        # H = ((H.T * mask).T)
        h11 = ((np.array([[-1,1]]).repeat(2 ** site_index, axis=0)).repeat(2 ** (num_site - site_index), axis=1)).reshape(-1)
        h00 = np.array([-1, 1]).repeat(2 ** num_site)
        mask = sparse.coo_matrix((h00 * h11 + 1) / 2)
        H = ((H.T).multiply(mask)).T

        return H

    if mode == 'BHalf':
        # H = B.detach().clone()
        H = BHalf.copy()
        for i in range(site_index):
            H = sparse.kron(I2, H, format='coo')
            # H = torch.kron(I2, H)

        for i in range(site_index + 1, num_site):
            H = sparse.kron(I2, H, format='coo')
            # H = torch.kron(H, I2)
        # H = torch.kron(H, B)
        # H = sparse.kron(H, BHalf, format='coo')
        # for i in range(num_site - site_index, num_site):
        #     H = sparse.kron(H, I2, format='coo')
            # H = torch.kron(H, I2)

        return H

    pass




#%%

def learn_init_state(num_site, net_depth, beta, batch_size, num_epoch, l_r, my_type, my_device, save_path, memo='init'):
    model = NAQS(num_site=num_site, net_depth=net_depth, my_type=my_type, my_device=my_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=l_r)

    TS = TensorSampling(num_site, beta, my_type, my_device)
    psi0, log_norm = TS.psi0()
    loss_func = lambda samples, beta: - TS.phi_func(psi0, samples) / model.psi_total(samples)

    model_out = learning(model, optimizer, loss_func, num_epoch=num_epoch, batch_size=batch_size, my_device=my_device, memo=memo, save_path=save_path)

    torch.save(model_out, save_path + memo + '_model_' + '.pth')
    torch.save(log_norm, save_path + memo + '_log_norm_' + '.pth')

    return model_out, log_norm

#%%
def eat_state_(num_site, net_depth, beta, batch_size, num_epoch, l_r, model0, my_type, my_device,save_path, memo='B_'):
    log_norm = 0
    model0 = model0.to(my_device)

    HN = HonNN(num_site, beta=beta, my_type=my_type, my_device=my_device)
    for site_index in range(num_site):
        # site_indexes = [site_index, num_site - site_index - 1]
        model = NAQS(num_site=num_site, net_depth=net_depth, my_type=my_type, my_device=my_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=l_r)

        H = calc_H(num_site, site_index, mode='B', beta=beta, my_type=my_type, my_device=my_device)
        norm = HN.calc_norm(H, model0)
        log_norm += torch.log(norm)
        loss_func = lambda samples, beta: - HN.B_on_site(site_index, model0.psi_total, samples,
                                                    beta=beta) / norm / model.psi_total(samples)

        model0 = learning(model, optimizer, loss_func, num_epoch=num_epoch, batch_size=batch_size,
                          my_device=my_device, memo=memo + '_' + str(site_index), save_path=save_path)

        torch.save(model0, save_path + memo + '_model_' + str(site_index) + '.pth')
        torch.save(torch.log(norm), save_path + memo + '_log_norm_' + str(site_index) + '.pth')

    return model0, log_norm


#%%
def eat_state(num_site, net_depth, beta, batch_size, num_epoch, l_r, model0, my_type, my_device, save_path, memo='eat'):
    log_norm = 0
    model0 = model0.to(my_device)
    HN = HonNN(num_site, beta=beta, my_type=my_type, my_device=my_device)

    # edge
    H = calc_H(num_site, 0, mode='edge', beta=beta, my_type=my_type, my_device=my_device)
    norm = HN.calc_norm(H, model0)
    log_norm += torch.log(norm)
    model = NAQS(num_site=num_site + 1, net_depth=net_depth, my_type=my_type, my_device=my_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
    loss_func = lambda samples, beta: - HN.Bs_on_edge(model0.psi_total, samples, beta=beta) / norm / model.psi_total(samples)
    model_out = learning(model, optimizer, loss_func, num_epoch=num_epoch, batch_size=batch_size, my_device=my_device, memo=memo + '_' + str(0), save_path=save_path)

    # model_saving_path = './model/'
    # data_saving_path = './data/'
    # torch.save(model_out, model_saving_path + 'model_%d_0_%d_.pth' % (num_site, 0))
    # torch.save(torch.log(norm), data_saving_path + 'log_norm_%d_0_%d_.pth' % (num_site, 0))

    torch.save(model_out, save_path + memo + '_model_' + str(0) + '.pth')
    torch.save(torch.log(norm), save_path + memo + '_log_norm_' + str(0) + '.pth')

    # bound
    model0 = model_out
    for site_index in range(3, num_site):
        H = calc_H(num_site, site_index, mode='bound', beta=beta, my_type=my_type, my_device=my_device)
        norm = HN.calc_norm(H, model0)
        log_norm += torch.log(norm)
        model = NAQS(num_site=num_site+1, net_depth=net_depth, my_type=my_type, my_device=my_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
        loss_func = lambda samples, beta: - HN.Bs_on_bound(model0.psi_total, site_index, samples, beta=beta) / norm / model.psi_total(samples)
        model0 = learning(model, optimizer, loss_func, num_epoch=num_epoch, batch_size=batch_size, my_device=my_device, memo=memo + '_' + str(site_index-2), save_path=save_path)

        # model_saving_path = './model/'
        # data_saving_path = './data/'
        # torch.save(model0, model_saving_path + 'model_%d_0_%d_.pth' % (num_site, site_index))
        # torch.save(torch.log(norm), data_saving_path + 'log_norm_%d_0_%d_.pth' % (num_site, site_index))

        torch.save(model0, save_path + memo + '_model_' + str(site_index - 2) + '.pth')
        torch.save(torch.log(norm), save_path + memo + '_log_norm_' + str(site_index - 2) + '.pth')

    # right edge
    H = calc_H(num_site, num_site, mode='bound', beta=beta, my_type=my_type, my_device=my_device)
    norm = HN.calc_norm(H, model0)
    log_norm += torch.log(norm)
    model = NAQS(num_site=num_site, net_depth=net_depth, my_type=my_type, my_device=my_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
    loss_func = lambda samples, beta: - HN.Bs_on_edge_r(model0.psi_total, samples, beta=beta) / norm / model.psi_total(samples)
    model_out = learning(model, optimizer, loss_func, num_epoch=num_epoch, batch_size=batch_size, my_device=my_device, memo=memo + '_' + str(num_site-2), save_path=save_path)

    # model_saving_path = './model/'
    # data_saving_path = './data/'
    # torch.save(model_out, model_saving_path + 'model_%d_1.pth' % (num_site))
    # torch.save(torch.log(norm), data_saving_path + 'log_norm_%d_1.pth' % (num_site))

    torch.save(model_out, save_path + memo + '_model_' + str(num_site-2) + '.pth')
    torch.save(torch.log(norm), save_path + memo + '_log_norm_' + str(num_site-2) + '.pth')


    return model_out, log_norm


def tail_on_state(num_site, net_depth, beta, batch_size, num_epoch, l_r, model0, my_type, my_device, save_path, memo='tail'):
    log_norm = 0
    model0 = model0.to(my_device)

    HN = HonNN(num_site, beta=beta, my_type=my_type, my_device=my_device)
    for site_index in range(num_site):
        # site_indexes = [site_index, num_site - site_index - 1]
        model = NAQS(num_site=num_site, net_depth=net_depth, my_type=my_type, my_device=my_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=l_r)

        H = calc_H(num_site, site_index, mode='BHalf', beta=beta, my_type=my_type, my_device=my_device)
        norm = HN.calc_norm(H, model0)
        log_norm += torch.log(norm)
        loss_func = lambda samples, beta: - HN.BHalf_on_site(site_index, model0.psi_total, samples,
                                                    beta=beta) / norm / model.psi_total(samples)

        model0 = learning(model, optimizer, loss_func, num_epoch=num_epoch, batch_size=batch_size,
                          my_device=my_device,memo=memo+'_'+str(site_index), save_path=save_path)

        # model_saving_path = './model/'
        # data_saving_path = './data/'
        # torch.save(model0, model_saving_path + 'model_%d_1_%d__.pth' % (num_site, site_index))
        # torch.save(torch.log(norm), data_saving_path + 'log_norm_%d_1_%d__.pth' % (num_site, site_index))

        torch.save(model0, save_path + memo + '_model_' + str(site_index) + '.pth')
        torch.save(torch.log(norm), save_path + memo + '_log_norm_' + str(site_index) + '.pth')

    return model0, log_norm



#%%
import os

# parameters
num_site = 4
net_depth = 1
beta = 1.0
# site_index = 4
# site_indexes = [site_index, num_site - site_index - 1]
my_type = torch.float64
my_device = torch.device('cuda:0')

batch_size = 1000
num_epoch = 20000
l_r = 0.001

# l_r_max = 0.01
# l_r_min = 0.0001
# lr_anneal = 1e-4



#%%
for num_site in [4]:
    for net_depth in [2,3]:
        model_saving_path = './model/' + 'trial0/' + 'n%d_d%d_b%d_lr%f/' % (num_site, 1, 1000, 0.001)
        # data_saving_path = './data/' + 'trial0/' + 'n%d_d%d_b%d_lr%f/' % (num_site, 1, batch_size, l_r)

        model0 = torch.load(model_saving_path + 'init' + '_model_' + '.pth')
        log_norm0 = torch.load(model_saving_path + 'init' + '_log_norm_' + '.pth')

        model_saving_path = './model/' + 'trial1/' + 'n%d_d%d_b%d_lr%f/' % (num_site, net_depth, batch_size, l_r)
        data_saving_path = './data/' + 'trial1/' + 'n%d_d%d_b%d_lr%f/' % (num_site, net_depth, batch_size, l_r)

        if not os.path.exists(model_saving_path):
            # os.mkdir(model_saving_path)
            os.makedirs(model_saving_path)

        if not os.path.exists(data_saving_path):
            # os.mkdir(data_saving_path)
            os.makedirs(data_saving_path)
        # model, log_norm = learn_init_state(num_site, net_depth, beta, batch_size, num_epoch, l_r, my_type, my_device)
        # model0 = torch.load(model_saving_path + 'model_%d_0.pth' % num_site)
        # model0 = torch.load(model_saving_path + 'model_%d_0_%d.pth' % (num_site, num_site // 2 - 1))
        # model, log_norm = eat_state_(num_site, net_depth, beta, batch_size, num_epoch, l_r, model0, my_type, my_device)
        # model, log_norm = eat_state(num_site, net_depth, beta, batch_size, num_epoch, l_r, model0, my_type, my_device)

        # torch.save(model, model_saving_path + 'model_%d_0.pth' % num_site)
        # torch.save(log_norm, data_saving_path + 'log_norm_%d_0.pth' % num_site)
        # log_norm = 0
        # model0, log_norm0 = learn_init_state(num_site, net_depth, beta, batch_size, num_epoch, l_r, my_type, my_device,
        #                                      save_path=data_saving_path)

        # print(log_norm0)
        # log_norm += log_norm0
        #
        eat_state_(num_site, net_depth, beta, batch_size, num_epoch, l_r, model0, my_type, my_device, save_path=data_saving_path)

        # log_norm += log_norm0
        # model0, log_norm0 = eat_state(num_site, net_depth, beta, batch_size, num_epoch, l_r, model0, my_type, my_device,
        #                               save_path=data_saving_path)
        # log_norm += log_norm0
        # model0, log_norm0 = tail_on_state(num_site, net_depth, beta, batch_size, num_epoch, l_r, model0, my_type, my_device,
        #                                   save_path=data_saving_path)
        # log_norm += log_norm0
        # print(log_norm * 2)





















