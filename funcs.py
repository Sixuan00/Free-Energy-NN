import torch
import numpy as np
from scipy.linalg import sqrtm
from scipy import sparse
from itertools import combinations


def calc_H(n, indexes, mode, beta=1.0):
    '''

    :param n: 总共的site数量
    :param indexes: 后面的比前面的大
    :param mode:
    :param beta:
    :return:
    '''

    B = np.array([[np.exp(beta), np.exp(-beta)], [np.exp(-beta), np.exp(beta)]])
    BHalf = sqrtm(B)
    I2 = np.eye(2)

    B = sparse.coo_matrix(B)
    BHalf = sparse.coo_matrix(BHalf)
    I2 = sparse.coo_matrix(I2)

    if mode == 'B':
        if len(indexes) == 1:
            H = B.copy()
            for i in range(indexes[0]):
                H = sparse.kron(I2, H, format='coo')
            for i in range(indexes[0] + 1, n):
                H = sparse.kron(H, I2, format='coo')
            return H
        H = B.copy()
        for i in range(indexes[0]):
            H = sparse.kron(I2, H, format='coo')
            # H = torch.kron(I2, H)

        for i in range(indexes[0] + 1, indexes[1]):
            H = sparse.kron(H, I2, format='coo')
            # H = torch.kron(H, I2)

        H = sparse.kron(H, B, format='coo')

        for i in range(indexes[1] + 1, n):
            H = sparse.kron(H, I2, format='coo')

        return H

    if mode == 'edge':
        h00 = ((np.array([[-1, 1]]).repeat(2 ** indexes[0], axis=0)).repeat(2 ** (n - indexes[0]-1), axis=1)).reshape(-1)
        h11 = ((np.array([[-1, 1]]).repeat(2 ** indexes[1], axis=0)).repeat(2 ** (n - indexes[1]-1), axis=1)).reshape(-1)

        mask = sparse.coo_array((h00 * h11 + 1) / 2)
        H = sparse.diags((np.exp(beta) * mask.todense() + np.exp(-beta) * (1 - mask.todense())).reshape(-1), format='coo')
        # B0 = np.array([[np.exp(beta), np.exp(-beta)], [np.exp(-beta), np.exp(beta)]])
        # H = sparse.diags(B0.reshape(-1).repeat(2 ** (num_site - 2)), format='coo')
        return H

    if mode == 'bound':
        H = B.copy()
        for i in range(indexes[0]):
            H = sparse.kron(I2, H, format='coo')

        for i in range(indexes[0] + 1, n):
            H = sparse.kron(H, I2, format='coo')

        h00 = ((np.array([[-1, 1]]).repeat(2 ** indexes[0], axis=0)).repeat(2 ** (n - indexes[0]-1), axis=1)).reshape(-1)
        h11 = ((np.array([[-1, 1]]).repeat(2 ** indexes[1], axis=0)).repeat(2 ** (n - indexes[1]-1), axis=1)).reshape(-1)

        mask = sparse.coo_matrix((h00 * h11 + 1) / 2)
        H = ((H.T).multiply(mask)).T

        return H


class HonNN:

    def __init__(self, num_site=4, beta=1.0, my_type=torch.float64, my_device=torch.device('cuda:1')):
        self.b = beta
        self.n = num_site
        self.type = my_type
        self.device = my_device

    '''

    :param psi_func: 
    :param mode: 
    :param s: 
    :param indexes: 作用上去的bond在现在的sample中的index
    :param beta: 
    :param my_type: 
    :param my_device: 
    :return: 
    '''

    def B_on_site(self, psi_func, index, s, beta=1.0):
        beta = torch.tensor([beta], dtype=self.type, device=self.device)
        mask = (s[:, index] + 1) / 2
        s1 = s.detach().clone()
        s_1 = s.detach().clone()
        s1[:, index] = torch.ones(s1.shape[0], dtype=self.type, device=self.device)
        s_1[:, index] = - torch.ones(s_1.shape[0], dtype=self.type, device=self.device)

        psi1 = torch.exp(beta) * psi_func(s1) + torch.exp(-beta) * psi_func(s_1)
        psi_1 = torch.exp(-beta) * psi_func(s1) + torch.exp(beta) * psi_func(s_1)

        psi = psi1 * mask + psi_1 * (1 - mask)

        return psi

    def B_on_edge_3d(self, psi_func, indexes, list_i0, s, beta=1.0):
        '''
        tested
        :param psi_func:
        :param indexes: sample里面要去掉的indexes,包括0和两个加上去的位点
        :param indexes0: 数值需要相同的indexes的list，list中每个list首位是位点的index
        :param s:
        :param beta:
        :return:
        '''
        beta = torch.tensor([beta], dtype=self.type, device=self.device)

        m = s.shape[1]

        index_list = [i for i in range(m)]
        for i in indexes:
            index_list.remove(i)
        # print(index_list)
        psi0 = psi_func(s[:, index_list])
        for indexes in list_i0:
            for i, j in combinations(indexes, 2):
                mask = (s[:, i] * s[:, j] + 1) / 2
                psi0 *= mask

        mask = (s[:, list_i0[0][0]] * s[:, list_i0[1][0]] + 1) / 2
        psi = (torch.exp(beta) * psi0 * mask + torch.exp(-beta) * psi0 * (1 - mask))

        return psi

    def Bs_on_bound_3d(self, psi_func, indexes, index, list_i0, s, beta=1.0):

        m = s.shape[1]
        index_list = [i for i in range(m)]

        for i in indexes:
            index_list.remove(i)

        psi = self.B_on_site(psi_func, index, s[:, index_list], beta=1.0)
        for indexes in list_i0:
            for i, j in combinations(indexes, 2):
                mask = (s[:, i] * s[:, j] + 1) / 2
                psi *= mask

        return psi

    def B_on_edge_r_3d(self, psi_func, list_i0, s, beta=1.0):
        s = torch.cat([(s[:, -1]).unsqueeze(1), s[:, :-1]], dim=1)
        psi = self.B_on_site(psi_func, 0, s, beta=beta)

        # for indexes in list_i0:
        #     for i, j in combinations(indexes, 2):
        #         mask = (s[:, i] * s[:, j] + 1) / 2
        #         psi *= mask
        mask = (s[:, 0] * s[:, -1] + 1) / 2
        psi *= mask

        return psi

    def sample2index(self, samples):
        samples = ((samples.detach().cpu().numpy() + 1) / 2).astype(int)
        if len(samples.shape) == 2:
            N = samples.shape[1]
            weights = np.array([2 ** (N - i - 1) for i in range(N)])
            index = (samples * weights).sum(axis=1)
            return torch.from_numpy(index).to(dtype=self.type, device=self.device)
        elif len(samples.shape) == 1:
            N = samples.shape[0]
            weights = np.array([2 ** (N - i - 1) for i in range(N)])
            index = (samples * weights).sum()
            return torch.tensor([index], dtype=self.type, device=self.device)

    def index2sample(self, index, num_site):
        samples = torch.zeros(index.shape[0], num_site, dtype=self.type, device=self.device)
        for i in range(num_site):
            samples[:, i] = index // (2 ** (num_site - i - 1))
            index %= 2 ** (num_site - i - 1)
        samples = samples * 2 - 1
        return samples

    def calc_norm(self, H, num_site, psi_func):

        A  = sparse.coo_matrix((H.T) @ H)
        r = torch.from_numpy(A.row).type(self.type).to(self.device)
        c = torch.from_numpy(A.col).type(self.type).to(self.device)
        s = self.index2sample(r, num_site)
        s_ = self.index2sample(c, num_site)
        Av = torch.from_numpy(A.data).type(self.type).to(self.device)

        norm = ((psi_func(s)).conj() * Av * psi_func(s_)).sum()
        return norm ** 0.5











