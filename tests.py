import kacward
import numpy as np
from scipy.linalg import sqrtm
from lnZ_2d import TensorSampling
import torch

# num_site = 4
# beta = 1.0
# beta = torch.tensor(beta, dtype=torch.float64, device='cuda:1')
# # b = torch.exp(torch.tensor(beta))
# B = torch.tensor([[torch.exp(beta), torch.exp(-beta)], [torch.exp(-beta), torch.exp(beta)]], dtype=torch.float64, device='cuda:1')
# H = B.detach().clone()
#
# for i in range(num_site - 1):
#     H = torch.kron(H, B)
#
# TS = TensorSampling(num_site, beta=1.0)
# # lnZ_exact = kacward.lnZ_2d_ferro_Ising(num_site, beta)
# # print(lnZ_exact)
#
# def Hamiltonian(num_site=4, beta=1.0, my_type=torch.float64, device=torch.device('cuda:1')):
#     beta = torch.tensor(beta, dtype=my_type, device=device)
#     B = torch.tensor([[torch.exp(beta), torch.exp(-beta)], [torch.exp(-beta), torch.exp(beta)]], dtype=my_type, device=device)
#     # B = torch.from_numpy(B).type(my_type).to(device)
#     # B = torch.tensor(sqrtm(B), dtype=my_type, device=device)
#     I3 = torch.zeros(2 ** 3, dtype=my_type, device=device)
#     I3[0], I3[-1] = 1, 1
#     I3 = I3.view(2, 2, 2)
#
#     I4 = torch.zeros(2 ** 4, dtype=my_type, device=device)
#     I4[0], I4[-1] = 1, 1
#     I4 = I4.view(2, 2, 2, 2)
#
#     # H3 = torch.einsum('ijk, il, jm, kn -> lmn', I3, B, B, B)
#     # H4 = torch.einsum('ijks, il, jm, kn, sq -> lmnq', I4, B, B, B, B)
#
#     H = I3.detach().clone()
#     for i in range(1, num_site-1):
#         # print(i)
#         # print(H.shape, B.shape)
#         H = torch.einsum('ijk, lj -> ilk', H, B)
#
#         H = torch.einsum('ijk, lmnj -> ilmkn', H, I4).contiguous().view(2 ** (i + 1), -1, 2 ** (i + 1))
#         # print(H.shape)
#
#     H = torch.einsum('ijk, lj -> ilk', H, B)
#     # print(H.shape)
#     H = torch.einsum('ijk, lmj -> ijkm', H, I3).contiguous().view([2] * num_site * 2)
#
#     # H = torch.einsum('ijk, lmj -> ilkm', H, H3).contiguous().view([2] * (lattice_size * 2))
#     # H = torch.einsum('ijklmnsq, ia, jb, lc, md -> abcdmnsq', H, B, B, B, B)
#
#     return H
#
#
# psi0, log_norm = TS.psi0()
# psi0 = psi0.view(-1)
# psi = H.to(dtype=torch.float64, device='cuda:1') @ psi0
# norm = torch.sqrt(torch.sum(psi**2))
# psi /= norm
#
# H = Hamiltonian(num_site, beta, my_type=torch.float64, device='cuda:1')
# phi = H.view(num_site ** 2, num_site ** 2) @ psi
# print(torch.log(torch.sqrt(torch.sum(phi**2))))
# # print(log_norm)
# # print(torch.log(torch.sqrt(torch.sum(psi**2))))


# with open('./data/loss0.txt')


from lnZ_2d import TensorSampling, HonNN
from scipy import sparse




def calc_H(num_site, site_index, layer_index, mode='B', beta=1.0, my_type=torch.float64, my_device=torch.device('cuda:1')):
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

        real_site_index = layer_index * num_site + 2 * site_index + 1
        real_num_site = (layer_index + 1) * num_site + site_index + 1

        for i in range(real_num_site):
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

        h11 = ((np.array([[-1,1]]).repeat(2 ** real_site_index, axis=0)).repeat(2 ** (real_num_site - real_site_index), axis=1)).reshape(-1)
        h00 = np.array([-1, 1]).repeat(2 ** real_num_site)
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




model_saving_path = './model/' + 'trial0/' + 'n%d_d%d_b%d_lr%f/' % (4, 1, 1000, 0.001)
model0 = torch.load(model_saving_path + 'init' + '_model_' + '.pth')
model0 = model0.to('cuda:1')
TS = TensorSampling(num_site=4, beta=1.0, my_device='cuda:1')
HN = HonNN(num_site=4, beta=1.0, my_device='cuda:1')


samples = TS.all_sample(num_site=5)
H = calc_H(num_site=4, layer_index=0, site_index=2, beta=1.0, mode='bound')
# H = H.to('cuda:1')
norm0 = HN.calc_norm(H, model0)
psi = HN.Bs_on_bound_3d(model0.psi_total, 2, samples)
print(torch.sqrt((psi * psi).sum()))

print(norm0)