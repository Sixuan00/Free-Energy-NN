from torch import nn
import torch
torch.set_default_dtype(torch.float64)


def intial_state(num_site=4, seed=1):
    torch.manual_seed(seed)
    model = NAQS(num_site)
    return model


class MaskedChannelLinear(nn.Linear):
    def __init__(self, input_size, channel_size, exclusive=False, bias=True):
        super(MaskedChannelLinear, self).__init__(input_size * channel_size, input_size * channel_size, bias)
        self.n = input_size
        self.nc = channel_size
        self.register_buffer('mask', torch.ones([input_size] * 2))
        if exclusive:
            self.mask = 1 - torch.triu(self.mask)
        else:
            self.mask = torch.tril(self.mask)

        self.mask = torch.kron(torch.eye(self.nc), self.mask)

        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, input):
        return nn.functional.linear(input, self.weight * self.mask, self.bias)


class NAQS(nn.Module):
    def __init__(self, num_site, num_channel=2, epsilon=1e-12, net_depth=1, my_type=torch.float32, my_device=torch.device('cuda:2')):
        super(NAQS, self).__init__()
        self.type = my_type
        self.device = my_device
        self.n = num_site
        self.nc = num_channel
        self.epsilon = epsilon

        self.layers = []
        self.layers.append(MaskedChannelLinear(self.n, self.nc, exclusive=True))
        for i in range(net_depth - 1):
            self.layers.append(self.build_block())
        self.layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.layers)

        self.register_buffer('psi_mask', torch.ones(self.n))
        self.psi_mask[0] = 0
        self.register_buffer('psi_bias', torch.zeros(self.n))
        self.psi_bias[0] = 0.5 ** 0.5


    def build_block(self):
        layers = []
        layers.append(nn.PReLU(self.n * self.nc))
        layers.append(MaskedChannelLinear(self.n, self.nc, exclusive=False))
        block = nn.Sequential(*layers)
        return block


    def forward(self, samples):
        psi = self.net(torch.cat([samples] * self.nc, dim=1))
        psi, theta = (psi.transpose(-1, 0)[:self.n, :]).transpose(-1,0), (psi.transpose(-1, 0)[self.n:, :]).transpose(-1,0)
        return psi * self.psi_mask + self.psi_bias, theta

    def psi_i(self, samples):
        psi, theta = self.forward(samples)
        psi, psi_ = psi, (1 - psi ** 2) ** 0.5 * torch.exp(1j * theta * 2 * torch.pi)
        return psi, psi_

    def sample(self, batch_size):
        samples = torch.ones([batch_size, self.n], dtype=self.type, device=self.device)
        for i in range(self.n):
            psi, theta = self.forward(samples)
            samples[:, i] = torch.bernoulli(psi[:, i] ** 2) * 2 - 1

        return samples

    def psi_total(self, samples):
        mask = (samples + 1) / 2
        psi, theta = self.forward(samples)
        log_psi = torch.log(psi + self.epsilon) * mask + (0.5 * torch.log(1 - psi * psi + self.epsilon)) * (1 - mask)
        # theta_total = (torch.sum(theta * (1 - mask), dim=1) % 1) * 2 * torch.pi
        return torch.exp(log_psi.sum(dim=1))

    def log_psi_conj(self, samples):
        mask = (samples + 1) / 2
        psi, theta = self.forward(samples)
        log_psi = torch.log(psi + self.epsilon) * mask + (0.5 * torch.log(1-psi*psi + self.epsilon)) * (1 - mask)
        log_theta = ((theta * (1 - mask)).sum(dim=1) % 1) * 2 * torch.pi
        return log_psi.sum(dim=1) - 1j * log_theta


def test():
    model = NAQS(10)
    print(model.layers[0].weight)

if __name__ == '__main__':
    test()