import torch
from NAQS import NAQS, intial_state
from lnZ_2d import TensorSampling, HonNN
import numpy as np
from scipy.linalg import sqrtm
from scipy import sparse
import csv
import time
def learning(model, model0, optimizer, loss_func, epoch_start, epoch_end, batch_size, my_device, beta=1.0, memo='init', save_path='./data/', B_anneal = 0.999, init_para=False):
    accuracy = 1
    model = model.to(my_device)
    # model_optim = model

    if init_para:
        model.load_state_dict(model0.state_dict())

    # if model.n == model0.n:
    #     if len(model.layers) == len(model0.layers):
    #         model.load_state_dict(model0.state_dict())
    #     else:
    #         model.layers[0].weight.data = model0.layers[0].weight.detach().clone()
    #         model.layers[0].bias.data = model0.layers[0].bias.detach().clone()

    t0 = time.time()
    beta0 = beta
    for epoch in range(epoch_start + 1, epoch_end + 1):

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
                # acc = abs(loss0 + 1)
                # if epoch > 500:
                #     if acc < accuracy:
                #         accuracy = acc
                #         model_optim = model

            with open(save_path + memo +'.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, loss0,
                                 ])
            torch.save(model, save_path + memo + '_model_' + str(epoch) + '.pth')
            print('epoch= {}, checkpoint_loss= {}, time= {}'.format(epoch, loss0, t1-t0))

    return model




def learn_init_state(model0, num_site, net_depth, beta, batch_size, num_epoch, l_r, my_type, my_device, save_path, memo='init', B_anneal=0.999):
    model = NAQS(num_site=num_site, net_depth=net_depth, my_type=my_type, my_device=my_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=l_r)

    TS = TensorSampling(num_site, beta, my_type, my_device)
    psi0, log_norm = TS.psi0()
    # loss_func = lambda samples, beta: - TS.phi_func(psi0, samples) / model.psi_total(samples)
    loss_func = lambda samples, beta: - TS.psi_func(psi0, samples) / model.psi_total(samples)

    es, ee = 0, num_epoch // 2
    model_out = learning(model=model, optimizer=optimizer, loss_func=loss_func, model0=model0, beta=beta, epoch_start=es,
                         epoch_end=ee, batch_size=batch_size, my_device=my_device, memo=memo, save_path=save_path)

    es, ee = num_epoch // 2, num_epoch
    model = NAQS(num_site=num_site, net_depth=net_depth, my_type=my_type, my_device=my_device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
    model_out = learning(model=model, optimizer=optimizer, loss_func=loss_func, model0=model_out, beta=beta,
                         epoch_start=es,
                         epoch_end=ee, batch_size=batch_size * 10, my_device=my_device, memo=memo, save_path=save_path, init_para=True)


    torch.save(model_out, save_path + memo + '_model_' + '.pth')
    torch.save(log_norm, save_path + memo + '_log_norm_' + '.pth')

    return model_out, log_norm

def init_base(psi_func, s, num_site):
    '''tested.'''
    psi = psi_func(s[:, :num_site])
    for i in range(num_site):
        mask = (s[:, i] * s[:, i + num_site] + 1) / 2
        psi *= mask
    return psi


def init_base_state(num_site, net_depth, beta, batch_size, num_epoch, l_r, my_type, my_device, save_path, memo='init_base', B_anneal=0.999):
    model = NAQS(num_site=num_site * 2, net_depth=net_depth, my_type=my_type, my_device=my_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=l_r)

    TS = TensorSampling(num_site, beta, my_type, my_device)
    psi0, log_norm = TS.psi0()
    psi_func = lambda s: TS.psi_func(psi0, s)
    loss_func = lambda samples, beta: - init_base(psi_func, samples, num_site) / model.psi_total(samples)

    es, ee = 0, num_epoch
    model_out = learning(model=model, optimizer=optimizer, loss_func=loss_func, model0=None, beta=beta, epoch_start=es, epoch_end=ee, batch_size=batch_size, my_device=my_device, memo=memo, save_path=save_path, B_anneal=B_anneal)

    torch.save(model_out, save_path + memo + '_model_' + '.pth')
    torch.save(log_norm, save_path + memo + '_log_norm_' + '.pth')

    return model_out, log_norm


def main():
    import os
    batch_size = 10000
    # l_r = 0.01
    beta_c = 0.44068679350977147
    beta = beta_c

    num_epoch = 4000
    my_type = torch.float64
    my_device = torch.device('cuda:4')
    # num_site = 6
    # net_depth = 3
    B_anneal = 0.95
    layer_i = 0
    mm = 'init'

    for num_site in [6]:
        for l_r in [0.1]:
            for net_depth in [1]:
                for batch_size in [1000]:
                    save_path = './data/' + '%s/' % mm + 'n%d_d%d_b%d_lr%f_a%f_ly%d/' % (
                        num_site, net_depth, batch_size, l_r, B_anneal, layer_i)

                    # model0 = torch.load(save_path + 'init_model_.pth')

                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    model0, log_norm0 = learn_init_state(
                        num_site=num_site,
                        net_depth=1,
                        beta=beta,
                        batch_size=batch_size,
                        num_epoch=num_epoch,
                        l_r=l_r,
                        my_type=my_type,
                        my_device=my_device,
                        save_path=save_path,
                        B_anneal=B_anneal,
                        model0=None
                    )

def test():
    import os
    batch_size = 1000
    # l_r = 0.01
    beta_c = 0.44068679350977147
    beta = beta_c

    num_epoch = 10000
    my_type = torch.float64
    my_device = torch.device('cuda:4')
    num_site = 6
    net_depth = 1
    B_anneal = 0.95
    layer_i = 0
    l_r = 0.08
    mm = 'init'
    save_path = './data/' + '%s/' % mm + 'n%d_d%d_b%d_lr%f_a%f_ly%d/' % (
        6, 1, batch_size, 0.1, B_anneal, layer_i)

    model = torch.load(save_path + 'init_model_.pth')
    save_path = './data/' + '%s/' % mm + 'n%d_d%d_b%d_lr%f_a%f_ly%d/' % (
        num_site, net_depth, batch_size, l_r, B_anneal, layer_i)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    init_base_state(num_site=num_site, net_depth=net_depth, beta=beta, batch_size=batch_size, num_epoch=num_epoch, l_r=l_r, my_type=my_type, my_device=my_device, save_path=save_path, memo='init', B_anneal=B_anneal)


def stitch_state(n,psi_func1, psi_func2, s, beta):
    sl = s[:, :n-1]
    sr = s[:, n-1:]
    beta = torch.tensor(beta, dtype=s.dtype, device=s.device)
    si = torch.ones(s.shape[0], dtype=s.dtype, device=s.device)
    si = si.unsqueeze(1)
    phi = torch.zeros([s.shape[0]], dtype=s.dtype, device=s.device)
    ilist = torch.tensor([-1, 1], dtype=s.dtype, device=s.device)
    for i in ilist:
        for j in ilist:
            s1 = torch.cat([i * si, sl], dim=1)
            s2 = torch.cat([sr, si * j], dim=1)
            phi += psi_func1(s1) * psi_func2(s2) * torch.exp(beta * i * j)
    return phi


def calc_H(num_site, beta):
    I = sparse.eye(2 ** (num_site - 2), format='coo')
    b1 = np.array([np.exp(beta), np.exp(-beta)])
    I1 = sparse.kron(I, b1, format='coo')
    b2 = np.array([np.exp(-beta), np.exp(beta)])
    I2 = sparse.kron(I, b2, format='coo')

    H = sparse.hstack([I1, I2], format='coo')

    return H





def marginal_psi_func(n,psi_func1, psi_func2, s):
    sl = s[:, : n]
    sr = s[:, n:]

    phi = psi_func1(sl) * psi_func2(sr)
    return phi


def test_stitch():
    import os
    batch_size = 1000
    # l_r = 0.01
    beta_c = 0.44068679350977147
    beta = beta_c

    num_epoch = 10000
    my_type = torch.float64
    my_device = torch.device('cuda:4')
    num_site = 6
    net_depth = 1
    B_anneal = 0.95
    layer_i = 0
    l_r = 0.08
    mm = 'init'
    save_path = './data/' + '%s/' % mm + 'n%d_d%d_b%d_lr%f_a%f_ly%d/' % (
        6, 1, batch_size, 0.1, B_anneal, layer_i)

    model1 = torch.load(save_path + 'init_model_.pth')

    save_path = './data/' + '%s/' % mm + 'n%d_d%d_b%d_lr%f_a%f_ly%d/' % (
        num_site, net_depth, batch_size, l_r, B_anneal, layer_i)

    model2 = torch.load(save_path + 'init_model_.pth')

    psi_func1 = model1.psi_total
    psi_func2 = model2.psi_total

    from lnZ_2d import TensorSampling
    TS = TensorSampling(num_site=num_site, beta=beta, my_type=my_type, my_device=my_device)
    s = TS.all_sample(num_site=16)
    # print(s.shape)
    phi = stitch_state(num_site, psi_func1, psi_func2, s, beta)
    print(torch.sqrt((phi*phi.conj()).sum()))

    H =calc_H(num_site=18, beta=beta)
    psi_func = lambda s: marginal_psi_func(num_site, psi_func1, psi_func2, s)

    from lnZ_2d import HonNN
    HN = HonNN(num_site=num_site, beta=beta, my_device=my_device, my_type=my_type)
    norm = HN.calc_norm(H, psi_func, n=num_site*3)
    print(norm)


    # norm =

def training(save_path, psi_func1, psi_func2, num_site=6, net_depth=1, my_type=torch.float64, my_device=torch.device('cuda:4'), batch_size=1000, num_epoch=10000, l_r=0.01, beta=0.44068679350977147, B_anneal=0.95, layer_i=0, memo='stitch'):
    model = NAQS(num_site=num_site * 3 - 2, net_depth=net_depth, my_type=my_type, my_device=my_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
    H = calc_H(num_site=num_site * 3, beta=beta)
    HN = HonNN(num_site=num_site, beta=beta, my_device=my_device, my_type=my_type)
    psi_func = lambda s: marginal_psi_func(num_site, psi_func1, psi_func2, s)
    norm = HN.calc_norm(H, psi_func, n=num_site * 3)
    log_norm = torch.log(norm)

    loss_func = lambda samples, beta: - stitch_state(num_site,psi_func1, psi_func2, samples, beta) / norm / model.psi_total(samples)

    es, ee = 0, num_epoch
    model_out = learning(model=model, optimizer=optimizer, loss_func=loss_func, model0=None, beta=beta, epoch_start=es,
                         epoch_end=ee, batch_size=batch_size, my_device=my_device, memo=memo, save_path=save_path,
                         B_anneal=B_anneal)

    torch.save(model_out, save_path + memo + '_model_' + '.pth')
    torch.save(log_norm, save_path + memo + '_log_norm_' + '.pth')

    return model_out, log_norm


def test_training():
    import os
    batch_size = 10000
    # l_r = 0.01
    beta_c = 0.44068679350977147
    beta = beta_c

    num_epoch = 20000
    my_type = torch.float64
    my_device = torch.device('cuda:4')
    num_site = 6
    net_depth = 2
    B_anneal = 0.95
    layer_i = 0
    l_r = 0.01
    mm = 'stitch'

    save_path = './data/' + '%s/' % 'init' + 'n%d_d%d_b%d_lr%f_a%f_ly%d/' % (
        6, 1, 1000, 0.1, B_anneal, layer_i)

    model1 = torch.load(save_path + 'init_model_.pth')

    save_path = './data/' + '%s/' % 'init' + 'n%d_d%d_b%d_lr%f_a%f_ly%d/' % (
        num_site, 1, 1000, 0.08, B_anneal, layer_i)

    model2 = torch.load(save_path + 'init_model_.pth')

    psi_func1 = model1.psi_total
    psi_func2 = model2.psi_total

    save_path = './data/' + '%s/' % mm + 'n%d_d%d_b%d_lr%f_a%f_ly%d/' % (
        num_site, net_depth, batch_size, l_r, B_anneal, layer_i)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    training(save_path, psi_func1, psi_func2, num_site=num_site, net_depth=net_depth, my_type=my_type, my_device=my_device, batch_size=batch_size, num_epoch=num_epoch, l_r=l_r, beta=beta, B_anneal=B_anneal, layer_i=layer_i, memo='stitch')


def _B_state(num_site, net_depth, beta, batch_size, num_epoch, l_r, model0, my_type, my_device, save_path, memo='_B_', B_anneal=0.999, i=0):
    log_norm = 0
    model0 = model0.to(my_device)
    HN = HonNN(num_site, beta=beta, my_type=my_type, my_device=my_device)

    n = model0.n
    # bound

    H = calc_H(n, beta=beta)
    norm = HN.calc_norm(H, model0.psi_total, n=n)
    log_norm += torch.log(norm)
    model = NAQS(num_site=n-2, net_depth=net_depth, my_type=my_type, my_device=my_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
    loss_func = lambda samples, beta: - HN.B_on_middle(model0.psi_total, samples, beta=beta) / norm / model.psi_total(samples)
    es, ee = 0, num_epoch
    model0 = learning(model, model0, optimizer, loss_func, beta=beta, epoch_start=es,epoch_end=ee, batch_size=batch_size,
                      my_device=my_device, memo=memo + '_' + str(i), save_path=save_path,
                      B_anneal=B_anneal)

    torch.save(model0, save_path + memo + '_model_' + str(i) + '.pth')
    torch.save(torch.log(norm), save_path + memo + '_log_norm_' + str(i) + '.pth')


    return model0, log_norm



def calc_lnZ():
    import os
    batch_size = 10000
    # l_r = 0.01
    beta_c = 0.44068679350977147
    beta = beta_c

    num_epoch = 20000
    my_type = torch.float64
    my_device = torch.device('cuda:4')
    num_site = 6
    net_depth = 2
    B_anneal = 0.95
    layer_i = 0
    l_r = 0.01
    mm = 'test0'

    save_path = './data/' + '%s/' % 'init' + 'n%d_d%d_b%d_lr%f_a%f_ly%d/' % (
        6, 1, 1000, 0.1, B_anneal, layer_i)

    model1 = torch.load(save_path + 'init_model_.pth')

    save_path = './data/' + '%s/' % 'init' + 'n%d_d%d_b%d_lr%f_a%f_ly%d/' % (
        num_site, 1, 1000, 0.08, B_anneal, layer_i)

    model2 = torch.load(save_path + 'init_model_.pth')

    psi_func1 = model1.psi_total
    psi_func2 = model2.psi_total



    log_norm = 0
    model0 = model1
    for j in range(num_site // 2 - 1):
        layer_i += j
        save_path = './data/' + '%s/' % mm + 'n%d_d%d_b%d_lr%f_a%f_ly%d/' % (
            num_site, net_depth, batch_size, l_r, B_anneal, layer_i)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        psi_func1 = model0.psi_total
        model0, log_norm0 = training(save_path, psi_func1, psi_func2, num_site=num_site, net_depth=net_depth,
                                     my_type=my_type,
                                     my_device=my_device, batch_size=batch_size, num_epoch=num_epoch, l_r=l_r,
                                     beta=beta, B_anneal=B_anneal,
                                     layer_i=layer_i, memo='stitch')

        log_norm += log_norm0

        for i in range(1, num_site):
            model0, log_norm0 = _B_state(num_site, net_depth, beta, batch_size, num_epoch, l_r, model0, my_type,
                                        my_device, save_path, memo='_B_', B_anneal=B_anneal, i=1)
            log_norm += log_norm0

    torch.save(log_norm, save_path + 'log_norm_final.pth')



if __name__ == '__main__':
    # main()
    # test()
    # test_stitch()
    # test_training()
    calc_lnZ()
    # test_stitch_2()
