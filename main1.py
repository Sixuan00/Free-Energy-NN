#%%
from funcs import calc_H, HonNN
from lnZ_2d import TensorSampling
import torch
import csv
import time
from NAQS import NAQS

#%%
model_saving_path = './model/' + 'trial0/' + 'n%d_d%d_b%d_lr%f/' % (4, 1, 1000, 0.001)
model0 = torch.load(model_saving_path + 'init' + '_model_' + '.pth')

# parameters
num_site = 4
net_depth = 1
beta = 1.0
my_type = torch.float64
my_device = torch.device('cuda:0')

batch_size = 10000
num_epoch = 20000
l_r = 0.01

#%%
def psi_init(s):
    '''tested.'''
    psi = model0.psi_total(s[:, :4])
    for i in range(num_site):
        mask = (s[:, i] * s[:, i + 4] + 1) / 2
        psi *= mask
    return psi

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

        if epoch % 1000 == 0:
            t1 = time.time()
            with torch.no_grad():
                samples0 = model.sample(batch_size * 100)
                loss0 = loss_func(samples0, beta)
                loss0 = loss0.mean().item()
            with open(save_path + memo +'.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, loss0,
                                 ])
            print('epoch= {}, checkpoint_loss= {}, time= {}'.format(epoch, loss0, t1-t0))

    return model
#%%
HN = HonNN(num_site=num_site, beta=beta, my_type=my_type, my_device=my_device)
TS = TensorSampling(num_site=num_site, beta=beta, my_type=my_type, my_device=my_device)

H = calc_H(8, [0,1], mode='bound')
norm = HN.calc_norm(H, 8, psi_init)
samples = TS.all_sample(8)
psi = HN.B_on_edge_r_3d(psi_init, [[4,5,6]], samples)
print(norm)
print(torch.sqrt((psi * psi).sum()))

#%%
def learn_init_state(num_site, net_depth, beta, batch_size, num_epoch, l_r, model0, my_type, my_device,save_path, memo='B_'):
    log_norm = 0

    for index in range(num_site, 2 * num_site):
        model = NAQS(num_site=num_site * 2, net_depth=net_depth, my_type=my_type, my_device=my_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
        indexes = [index]
        H = calc_H(num_site * 2, indexes, mode='B', beta=beta)
        if index == num_site:
            norm = HN.calc_norm(H, num_site * 2, psi_init)
        else:
            norm = HN.calc_norm(H, num_site * 2, model0.psi_total)
        log_norm += torch.log(norm)
        loss_func = lambda samples, beta: - HN.B_on_site(psi_init, index, samples,
                                                         beta=beta) / norm / model.psi_total(samples)

        model0 = learning(model, optimizer, loss_func, num_epoch, batch_size, my_device, beta=beta, memo=memo + str(index - 4), save_path=save_path)
        torch.save(model0, save_path + memo + '_model_' + str(index) + '.pth')
        torch.save(torch.log(norm), save_path + memo + '_log_norm_' + str(index) + '.pth')

    return model0, log_norm

#%%
import os
save_path = './data/' + 'trial1_3d/' + 'n%d_d%d_b%d_lr%f/' % (num_site, net_depth, batch_size, l_r)
if not os.path.exists(save_path):
    os.makedirs(save_path)
# model0 = learn_init_state(num_site, net_depth, beta, batch_size, num_epoch, l_r, model0, my_type, my_device, save_path=save_path, memo='B_')


#%%
filename = './data/' + 'trial1_3d/' + 'n%d_d%d_b%d_lr%f/' % (num_site, net_depth, batch_size, l_r) + 'B__model_7.pth'
model0 = torch.load(filename)
def learn_init_state1(num_site, net_depth, layer_i, beta, batch_size, num_epoch, l_r, model0, my_type, my_device,save_path, memo='eat'):
    log_norm = 0

    # model = NAQS(num_site=model0.n + 3, net_depth=net_depth, my_type=my_type, my_device=my_device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
    #
    # H = calc_H(model0.n, [num_site * layer_i, num_site * layer_i + 1], mode='edge')
    # norm = HN.calc_norm(H, model0.n, model0.psi_total)
    # log_norm += torch.log(norm)
    # indexes = [0, num_site * layer_i + 1, num_site * layer_i + 2]
    # list_i0 = [[num_site * layer_i + 1, num_site * layer_i + 3], [0, num_site * layer_i + 2, num_site * layer_i + 4]]
    # loss_func = lambda samples, beta: - HN.B_on_edge_3d(model0.psi_total, indexes, list_i0, samples,
    #                                                  beta=beta) / norm / model.psi_total(samples)
    #
    # model0 = learning(model, optimizer, loss_func, num_epoch, batch_size, my_device, beta=beta,
    #                   memo=memo + '_' + str(0), save_path=save_path)
    # torch.save(model0, save_path + memo + '_model_' + str(0) + '.pth')
    # torch.save(torch.log(norm), save_path + memo + '_log_norm_' + str(0) + '.pth')


    # for i in range(num_site - 3):
    #     model = NAQS(num_site=model0.n + 1, net_depth=net_depth, my_type=my_type, my_device=my_device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
    #     indexes = [0, num_site * layer_i + (2 + i) * 2 + 1]
    #     H = calc_H(model0.n, indexes, mode='bound')
    #     norm = HN.calc_norm(H, model0.n, model0.psi_total)
    #     log_norm += torch.log(norm)
    #     indexes = [num_site * layer_i + 3 + i]
    #     list_i0 = [[0, indexes[0], indexes[0] + 3 + i]]
    #     index = num_site * layer_i + (2 + i) * 2 + 1
    #     loss_func = lambda samples, beta: - HN.Bs_on_bound_3d(model0.psi_total, indexes, index, list_i0, samples,
    #                                                         beta=beta) / norm / model.psi_total(samples)
    #
    #     model0 = learning(model, optimizer, loss_func, num_epoch, batch_size, my_device, beta=beta,
    #                       memo=memo + '_' + str(i+1), save_path=save_path)
    #     torch.save(model0, save_path + memo + '_model_' + str(i+1) + '.pth')
    #     torch.save(torch.log(norm), save_path + memo + '_log_norm_' + str(i+1) + '.pth')

    model0 = torch.load(save_path + memo + '_model_' + str(0+1) + '.pth')
    print(model0.n)

    model = NAQS(num_site=model0.n, net_depth=net_depth, my_type=my_type, my_device=my_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=l_r)

    indexes = [0, num_site * (layer_i + 2) - 1]
    H = calc_H(model0.n, indexes, mode='bound')
    norm = HN.calc_norm(H, model0.n, model0.psi_total)
    log_norm += torch.log(norm)
    index = num_site * (layer_i + 2) - 1
    list_i0 = [[num_site * (layer_i + 1) - 1, num_site * (layer_i + 2) - 1]]
    loss_func = lambda samples, beta: - HN.B_on_edge_r_3d(model0.psi_total, list_i0, samples,
                                                            beta=beta) / norm / model.psi_total(samples)

    model0 = learning(model, optimizer, loss_func, num_epoch, batch_size, my_device, beta=beta,
                      memo=memo + '_' + str(0), save_path=save_path)
    torch.save(model0, save_path + memo + '_model_' + str(num_site - 2) + '.pth')
    torch.save(torch.log(norm), save_path + memo + '_log_norm_' + str(num_site - 2) + '.pth')



    return model0, log_norm

layer_i = 1
learn_init_state1(num_site, net_depth, layer_i, beta, batch_size, num_epoch, l_r, model0, my_type, my_device,save_path, memo='eat')