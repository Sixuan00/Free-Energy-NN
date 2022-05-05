import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy import sparse

class TensorSampling:
    def __init__(self, num_site, beta=1.0, my_type=torch.float64, my_device=torch.device('cuda:1')):
        super(TensorSampling, self).__init__()
        self.n = num_site
        self.b = beta
        B = np.array([[np.exp(self.b), np.exp(-self.b)], [np.exp(-self.b), np.exp(self.b)]])
        self.B = torch.from_numpy(B).to(dtype=my_type, device=my_device)
        BHalf = torch.from_numpy(sqrtm(B)).type(my_type).to(my_device)
        I3 = torch.zeros(2 ** 3, dtype=my_type, device=my_device)
        I3[0], I3[-1] = 1, 1
        self.I3 = I3.view(2, 2, 2)

        I2 = torch.zeros(2 ** 2, dtype=my_type, device=my_device)
        I2[0], I2[-1] = 1, 1
        self.I2 = I2.view(2, 2)

        self.type = my_type
        self.device = my_device

    def psi0(self):
        psi0 = self.I2.detach().clone()

        for i in range(1, self.n - 1):
            psi0 = torch.einsum('ij,ki->kj', psi0, self.B)
            psi0 = torch.einsum('ij, kli->kjl', psi0, self.I3).contiguous().view(-1, 2 ** (i + 1))

        psi0 = torch.einsum('ij,ki->kj', psi0, self.B)
        psi0 = torch.einsum('ij, ki -> jk', psi0, self.I2).contiguous().view([2] * self.n)

        norm = torch.sqrt(torch.sum(psi0 * psi0))
        psi0 = psi0 / norm

        return psi0, torch.log(norm)



    def phi_func(self, phi, samples):
        phi0 = torch.zeros(samples.shape[0], dtype=self.type, device=self.device)
        phi = phi.view(-1)

        for i in range(samples.shape[0]):
            index = ((samples[i, :].cpu().numpy() + 1) / 2).astype(int)
            # print(index)
            index_str = ''.join(str(i) for i in index)
            index_reshape = int(index_str, 2)
            phi0[i] = phi[index_reshape]
        return phi0

    def all_sample(self, num_site=None):
        if num_site is None:
            num_site = self.n
        samples = torch.zeros([2 ** num_site, num_site], dtype=self.type, device=self.device)
        for i in range(2 ** num_site):
            samples[i, :] = torch.tensor([int(j) for j in bin(i)[2:].zfill(num_site)], dtype=self.type, device=self.device)
        return samples * 2 - 1



class HonNN:
    def __init__(self, num_site, beta=1.0, my_type=torch.float64, my_device=torch.device('cuda:1')):
        super(HonNN, self).__init__()
        self.n = num_site
        self.b = beta
        self.type = my_type
        self.device = my_device

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

    def B_on_site(self, site_index, psi_func, s, beta=None):
        if beta is None:
            beta = self.b
        beta = torch.tensor([beta], dtype=self.type, device=self.device)
        mask = (s[:, site_index] + 1) / 2
        s1 = s.detach().clone()
        s_1 = s.detach().clone()
        s1[:, site_index] = torch.ones(s1.shape[0], dtype=self.type, device=self.device)
        s_1[:, site_index] = - torch.ones(s_1.shape[0], dtype=self.type, device=self.device)

        psi1 = torch.exp(beta) * psi_func(s1) + torch.exp(-beta) * psi_func(s_1)
        psi_1 = torch.exp(-beta) * psi_func(s1) + torch.exp(beta) * psi_func(s_1)

        psi = psi1 * mask + psi_1 * (1 - mask)

        return psi


    def BHalf_on_site(self, site_index, psi_func, s, beta=None):
        if beta is None:
            beta = self.b
        # beta = torch.tensor([beta], dtype=self.type, device=self.device)
        B = np.array([[np.exp(beta), np.exp(-beta)], [np.exp(-beta), np.exp(beta)]])
        BHalf = torch.from_numpy(sqrtm(B)).type(self.type).to(self.device)
        mask = (s[:, site_index] + 1) / 2
        s1 = s.detach().clone()
        s_1 = s.detach().clone()
        s1[:, site_index] = torch.ones(s1.shape[0], dtype=self.type, device=self.device)
        s_1[:, site_index] = - torch.ones(s_1.shape[0], dtype=self.type, device=self.device)

        psi1 = BHalf[0,0] * psi_func(s1) + BHalf[0,1] * psi_func(s_1)
        psi_1 = BHalf[1,0] * psi_func(s1) + BHalf[1,1] * psi_func(s_1)

        psi = psi1 * mask + psi_1 * (1 - mask)

        return psi

    def B_on_sites(self, site_indexes, psi_func, s, beta=None):
        if beta is None:
            beta = self.b
        if len(site_indexes) == 0:
            return psi_func(s)
        return self.B_on_site(site_indexes[-1], lambda s: self.B_on_sites(site_indexes[:-1], psi_func, s, beta), s, beta)


    def BHalf_on_sites(self, site_indexes, psi_func, s, beta=None):
        if beta is None:
            beta = self.b
        if len(site_indexes) == 0:
            return psi_func(s)
        return self.BHalf_on_site(site_indexes[-1], lambda s: self.BHalf_on_sites(site_indexes[:-1], psi_func, s, beta), s, beta)

    def Bs_on_edge(self, psi_func, s, mode='0', beta=None):
        if beta is None:
            beta = self.b

        beta = torch.tensor([beta], dtype=self.type, device=self.device)
        if mode == '0':
            mask = (s[:, 1] * s[:, 2] + 1) / 2
            mask0 = (s[:, 0] * s[:, 2] + 1) / 2

            psi0 = psi_func(s[:, 1:])
            psi = (torch.exp(beta) * psi0 * mask + torch.exp(-beta) * psi0 * (1 - mask)) * mask0

            return psi

        if mode == '1':
            mask = (s[:, 1] * s[:, 2] + 1) / 2
            mask0 = (s[:, 0] * s[:, 2] + 1) / 2

            psi0 = psi_func(s)
            psi = (torch.exp(beta) * psi0 * mask + torch.exp(-beta) * psi0 * (1 - mask)) * mask0

            return psi

        if mode == '-1':
            mask = (s[:, -2] * s[:, -3] + 1) / 2
            mask0 = (s[:, -1] * s[:, -3] + 1) / 2

            psi0 = psi_func(s[:, 1:-1])
            psi = (torch.exp(beta) * psi0 * mask + torch.exp(-beta) * psi0 * (1 - mask)) * mask0

            return psi

    def Bs_on_edges(self, psi_func, s, beta=None):
        if beta is None:
            beta = self.b
        return self.Bs_on_edge(lambda s: self.Bs_on_edge(psi_func, s, '-1', beta), s, '1', beta)

    def B_on_middle(self, psi_func, s, beta=None):
        if beta is None:
            beta = self.b

        i1 = self.n // 2 - 1
        i2 = self.n // 2

        I = torch.ones(s.shape[0], dtype=self.type, device=self.device)
        s11 = torch.cat([(I).unsqueeze(1), s], dim=1)
        s11 = torch.cat([s11, (I).unsqueeze(1)], dim=1)
        s12 = torch.cat([(I).unsqueeze(1), s], dim=1)
        s12 = torch.cat([s12, (-I).unsqueeze(1)], dim=1)
        s21 = torch.cat([(-I).unsqueeze(1), s], dim=1)
        s21 = torch.cat([s21, (I).unsqueeze(1)], dim=1)
        s22 = torch.cat([(-I).unsqueeze(1), s], dim=1)
        s22 = torch.cat([s22, (-I).unsqueeze(1)], dim=1)



        # mask = (s[:, i1] * s[:, i2] + 1) / 2

        # psi0 = psi_func(s2)
        beta = torch.tensor([beta], dtype=self.type, device=self.device)
        psi = (torch.exp(beta) * (psi_func(s11) + psi_func(s22)) + torch.exp(-beta) * (psi_func(s12) + psi_func(s21)))
        # psi = self.B_on_site(0, psi_func, s2, beta=beta)
        # psi *= mask
        # psi = self.B_on_site(0, psi_func, s, beta=beta)

        return psi



    def Bs_on_bound(self, psi_func, site_index, s, beta=None):
        if beta is None:
            beta = self.b

        psi = self.B_on_site(0, psi_func, s, beta=beta)
        mask = (s[:, 0] * s[:, site_index] + 1) / 2
        psi *= mask

        return psi

    def Bs_on_edge_r(self, psi_func, s, beta=None):
        if beta is None:
            beta = self.b

        s = torch.cat([(s[:, -1]).unsqueeze(1), s], dim=1)
        psi = self.B_on_site(0, psi_func, s, beta=beta)

        return psi

    def calc_norm(self, H, model):
        # A = (H.conj().T) @ H
        # A = A.to_sparse()
        # Ai, Av = A.indices(), A.values()
        #
        # s = self.index2sample(Ai[0], model.n)
        # s_ = self.index2sample(Ai[1], model.n)
        A  = sparse.coo_matrix((H.T) @ H)
        r = torch.from_numpy(A.row).type(self.type).to(self.device)
        c = torch.from_numpy(A.col).type(self.type).to(self.device)
        s = self.index2sample(r, model.n)
        s_ = self.index2sample(c, model.n)
        Av = torch.from_numpy(A.data).type(self.type).to(self.device)

        norm = ((model.psi_total(s)).conj() * Av * model.psi_total(s_)).sum()
        return norm ** 0.5




if __name__ == '__main__':
    pass
