import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

def load_data(filename):
    data = pd.read_csv(filename)
    data = data.to_numpy()
    return data

def main():
    batch_size = 1000
    l_r = 0.001
    beta_c = 0.44068679350977147
    beta = beta_c
    B_anneal = 0.9985

    num_epoch = 30000
    my_type = torch.float64
    my_device = torch.device('cuda:4')
    num_site = 4
    net_depth = 3
    path = './data/' + 'ana_trial1/' + 'n%d_d%d_b%d_lr%f_a%f/' % (num_site, net_depth, batch_size, l_r, B_anneal)

    fig, ax = plt.subplots()
    memos = ['edge', '_B_']
    labels = ['edge_sym', 'B']
    for i in range(2):
        filename = path + memos[i] + '_0.csv'
        data = load_data(filename)
        ax.plot(data[:, 0], abs(data[:, 1] + 1), 'o-', label=labels[i])


    x_ticks = np.arange(0, num_epoch + 1, 5000)
    ax.set_yscale('log')
    ax.set_xticks(x_ticks)
    ax.legend()
    ax.set(xlabel='epoch', ylabel='|loss+1|')
    ax.set_title('lr={}, num_site={}, net_depth={}'.format(l_r, num_site, net_depth))
    ax.grid()


    plt.show()



def calc_log_norm():
    batch_size = 1000
    l_r = 0.001
    beta_c = 0.44068679350977147
    beta = beta_c
    num_site = 4
    num_epoch = 20000
    B_anneal = 0.9985
    my_type = torch.float64
    my_device = torch.device('cuda:4')

    log_norm = 0
    # log_norm = log_norm.to(my_device)
    path = './data/' + 'betac_trial1/' + 'n%d_d%d_b%d_lr%f/' % (num_site, 1, 1000, 0.001)
    log_norm0 = torch.load(path + 'init_' + 'log_norm_.pth')
    log_norm0 = log_norm0.to(my_device)
    log_norm += log_norm0
    # print(log_norm0)

    path = './data/' + 'ana_trial0/' + 'n%d_d%d_b%d_lr%f/' % (num_site, 3, 1000, 0.001)
    log_norm0 = torch.load(path + 'B_' + '_log_norm_0.pth')
    log_norm0 = log_norm0.to(my_device)
    log_norm += log_norm0

    log_norm0 = torch.load(path + 'B_' + '_log_norm_1.pth')
    log_norm0 = log_norm0.to(my_device)
    log_norm += log_norm0


    path = './data/' + 'ana_trial1/' + 'n%d_d%d_b%d_lr%f_a%f/' % (num_site, 3, 1000, 0.001, 0.9985)
    log_norm0 = torch.load(path + 'edge' + '_log_norm_0.pth')
    log_norm0 = log_norm0.to(my_device)
    log_norm += log_norm0

    log_norm0 = torch.load(path + '_B_' + '_log_norm_0.pth')
    log_norm0 = log_norm0.to(my_device)
    log_norm += log_norm0

    log_norm0 = torch.load(path + 'BHalf_' + '_log_norm_1.pth')
    log_norm0 = log_norm0.to(my_device)
    log_norm += log_norm0

    log_norm0 = torch.load(path + 'BHalf_' + '_log_norm_0.pth')
    log_norm0 = log_norm0.to(my_device)
    log_norm += log_norm0

    print(2 * log_norm)
    # path = './data/' + 'ana_trial1/' + 'n%d_d%d_b%d_lr%f_a%F/' % (num_site, 3, 1000, 0.001, 0.9985)


if __name__ == '__main__':
    # main()
    calc_log_norm()