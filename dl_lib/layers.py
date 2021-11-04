from fastai.vision.all import *

__all__ = ['SA']

class SA(Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels, norm_type=NormType.Batch):
        self.query = ConvLayer(n_channels, n_channels//8, ks=1, ndim=1, norm_type=norm_type, act_cls=None, bias=False)
        self.key   = ConvLayer(n_channels, n_channels//8, ks=1, ndim=1, norm_type=norm_type, act_cls=None, bias=False)
        self.value = ConvLayer(n_channels, n_channels,    ks=1, ndim=1, norm_type=norm_type, act_cls=None, bias=False)
        self.gamma = nn.Parameter(tensor([0.]))

    def forward(self, x):
        #Notation from the paper.=
        size = x.size()
        x = x.view(size[0], size[1],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(size[0], size[1], size[2], size[3]).contiguous()