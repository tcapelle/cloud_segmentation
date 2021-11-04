from fastai.vision.models.xresnet import XResNet, ResBlock
from fastai.vision.all import *
from fastai.vision.models.unet import UnetBlock

from ..layers import SA

import torchvision
if '0.9' in torchvision.__version__:
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

__all__ = ['SimpleUnet', 'simple_unet_split', 'DeepLabV3']

set_seed(2021)

    
class MyUnetBlock(Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."
    @delegates(ConvLayer.__init__)
    def __init__(self, up_in_c, x_in_c, final_div=True, blur=False, act_cls=defaults.activation,
                 self_attention=False, init=nn.init.kaiming_normal_, norm_type=None, **kwargs):
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, act_cls=act_cls, norm_type=norm_type)
        self.bn = BatchNorm(x_in_c)
        ni = up_in_c//2 + x_in_c
        nf = ni if final_div else x_in_c
        self.conv1 = ConvLayer(ni, nf, act_cls=act_cls, norm_type=norm_type, **kwargs)
        self.conv2 = ConvLayer(nf, nf, act_cls=act_cls, norm_type=norm_type,
                               xtra=SA(nf, norm_type) if self_attention else None, **kwargs)
        self.relu = act_cls()
        apply_init(nn.Sequential(self.conv1, self.conv2), init)
    
    def forward(self, up_in, s):
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))
    
class SimpleUnet(Module):
    def __init__(self, n_in=3, n_classes=7, act_cls=defaults.activation,
                 norm_type=NormType.Batch, sa=False, blur=True, l_szs = [32,48,96,128]):
        "A simpler UNET torch.jit compatible"
        self.stem = nn.Sequential(ConvLayer(n_in, 32, ks=3, stride=2, act_cls=act_cls),
                                  ConvLayer(32, 32, ks=3, stride=1, act_cls=act_cls),
                                  ConvLayer(32, l_szs[0], ks=3, stride=1, act_cls=act_cls),
                                  nn.MaxPool2d(3, stride=2, padding=1))
        
        
        assert len(l_szs)==4, '3 layers supported (4 values)'
        szs = zip(l_szs[0:-1], l_szs[1:])
        
        self.l1, self.l2, self.l3 = [self.make_resblock(ni,nf) for (ni,nf) in szs]
        
        ni = l_szs[-1]
        self.middle_conv = nn.Sequential(BatchNorm(ni), nn.ReLU(),
            ConvLayer(ni, ni * 2, act_cls=act_cls, norm_type=norm_type),
            ConvLayer(ni * 2, ni, act_cls=act_cls, norm_type=norm_type)
                                        )
        l_szs_r = l_szs[::-1]
        r_szs = zip(l_szs_r[0:-1], l_szs_r[1:])
        
        unet_blocks = []
        for i, (ni, nf) in enumerate(r_szs):
            unet_blocks += [MyUnetBlock(ni, nf, final_div=(i==len(l_szs)-2), act_cls=act_cls, 
                                       norm_type=norm_type, self_attention= (i==0) and sa, blur=blur)]
        self.u3, self.u2, self.u1 = unet_blocks
        final_n_filters = l_szs[0]+l_szs[1]//2
        self.shuffle = PixelShuffle_ICNR(final_n_filters, act_cls=act_cls, norm_type=norm_type)
        self.head = nn.Sequential(ResBlock(1, final_n_filters+n_in, 32, stride=1, 
                                           act_cls=act_cls,norm_type=norm_type), 
                                  ConvLayer(32, n_classes, ks=1, act_cls=None, norm_type=norm_type))
    @classmethod
    def unet_out_sz(up_in_c, x_in_c, final_div):
        ni = up_in_c//2 + x_in_c
        nf = ni if final_div else ni//2
        return nf
    
    def make_resblock(self, ni, nf):
        return nn.Sequential(ResBlock(1, ni, nf, stride=2), ResBlock(1, nf, nf, stride=1))
    
    def forward(self, x):
        
        #encode
        out = self.stem(x)
        out1 = self.l1(out) 
        out2 = self.l2(out1)
        out3 = self.l3(out2)
        
        #middle
        res = self.middle_conv(out3)
        
        #decode
        res = self.u3(res, out2)
        res = self.u2(res, out1)
        res = self.u1(res, out)
        
        #interp
        if res.shape[-2:] != x.shape[-2:]:
            res = self.shuffle(res)
            res = F.interpolate(res, x.shape[-2:], mode='nearest')
        res = torch.cat([x, res], dim=1)
        res = self.head(res)
        return res
    
def simple_unet_split(m): 
    return [*L(m.l1, m.l2, m.l3, m.middle_conv, m.u3, m.u2, m.u1).map(params), params(m.head)]




class DeepLabV3(Module):
    def __init__(self, n_classes):
        self.model = deeplabv3_mobilenet_v3_large(num_classes=n_classes)
    def forward(self, x):
        return self.model(x)['out']