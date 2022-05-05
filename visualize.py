import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data(filename):
    data = pd.read_csv(filename)
    data = data.to_numpy()
    return data


num_site = 16
net_depth = 1
beta = 1.0
# site_index = 4
# site_indexes = [site_index, num_site - site_index - 1]
my_type = torch.float64
my_device = torch.device('cuda:0')

batch_size = 1000
num_epoch = 20000
l_r = 0.001


# fig = plt.figure()
fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True,sharex=True)
for k, net_depth in enumerate([1,2,3,4]):
    model_saving_path = './model/' + 'test_trial5/' + 'n%d_d%d_b%d_lr%f/' % (num_site, net_depth, batch_size, l_r)
    data_saving_path = './data/' + 'test_trial5/' + 'n%d_d%d_b%d_lr%f/' % (num_site, net_depth, batch_size, l_r)

    save_path = data_saving_path

    # memo = 'init'
    # filename = save_path + memo +'.csv'

    # site_index = 1
    # memo = 'eat' + '_' + str(site_index)
    # filename = save_path + memo + '.csv'

    # filename_list = []

    # memos = ['init', 'eat__0', 'eat__1', 'eat_0', 'eat_1', 'eat_2', 'tail_0', 'tail_1']

    # fig, ax = plt.subplots()
    # ax = fig.add_subplot(1, 3, i+1)
    for i in range(num_site):
        memo = 'B__' + str(i)
        filename = save_path + memo + '.csv'
        data = load_data(filename)
        # ax.scatter(data[:, 0][:20], abs(data[:, 1][:20] + 1), label=memo)
        axs[k].plot(data[:, 0][:20], abs(data[:, 1][:20] + 1), 'o-', label=memo)
        axs[k].legend()

    # plt.show()
    # data = load_data(filename)
    # epoch, loss = data[:, 0], data[:, 1]

    # ax.plot(epoch, abs(loss + 1))
    axs[k].set_yscale('log')
    axs[k].set(xlabel='epoch', ylabel='|loss+1|')
    axs[k].set_title('num_site={}, net_depth={}'.format(num_site, net_depth))
    axs[k].grid()
plt.show()




