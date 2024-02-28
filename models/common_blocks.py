import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def fuse_conv_bn(conv, bn):

    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean) / var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                           conv.out_channels,
                           conv.kernel_size,
                           conv.stride,
                           conv.padding,
                           bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


class Conv(nn.Module):
    def __init__(
            self,
            c1,  # in channels
            c2,  # out channels 
            k=1,  # kernel size 
            s=1,  # stride
            p=None,  # padding
            d=1,  # dilation
            act_type='lrelu',  # activation
            depthwise=False,
            groups=1,
            bias=False,
            bn=False):
        super(Conv, self).__init__()
        self.convs = nn.Sequential()
        self.act_type = act_type
        self.bn = bn
        self.depthwise = depthwise
        self.quant = False
        act = self.get_activation()

        if depthwise:
            # depthwise conv
            self.convs.add_module(
                'conv1',
                nn.Conv2d(c1,
                          c1,
                          kernel_size=k,
                          stride=s,
                          padding=autopad(k, p),
                          dilation=d,
                          groups=c1,
                          bias=bias))
            self.convs.add_module('bn1', nn.BatchNorm2d(c1))
            self.convs.add_module('act1', act)
            # pointwise conv
            self.convs.add_module(
                'conv2',
                nn.Conv2d(c1,
                          c2,
                          kernel_size=1,
                          stride=s,
                          padding=0,
                          dilation=d,
                          groups=1,
                          bias=bias))
            self.convs.add_module('bn2', nn.BatchNorm2d(c2))
            self.convs.add_module('act2', act)

        else:
            self.convs.add_module(
                'conv',
                nn.Conv2d(c1,
                          c2,
                          kernel_size=k,
                          stride=s,
                          padding=autopad(k, p),
                          dilation=d,
                          groups=groups,
                          bias=bias))
            if self.bn:
                self.convs.add_module('bn', nn.BatchNorm2d(c2))
            self.convs.add_module('act', act)

    def forward(self, x):
        return self.convs(x)

    def get_activation(self):
        if self.act_type == 'relu':
            act = nn.ReLU(inplace=True)
        elif self.act_type == 'lrelu':
            act = nn.LeakyReLU(inplace=True)
        else:
            act = nn.Identity()
        return act

    def fuse_params(self):
        if (not self.quant) and hasattr(self.convs, 'conv') and hasattr(
                self.convs, 'bn'):
            convs = fuse_conv_bn(self.convs._modules['conv'],
                                 self.convs._modules['bn'])
            act = self.get_activation()
            self.convs = nn.Sequential()
            self.convs.add_module('conv', convs)
            self.convs.add_module('act', act)

    def get_fused_kernel(self):
        if (not self.quant) and hasattr(self.convs, 'conv') and hasattr(
                self.convs, 'bn'):
            convs = fuse_conv_bn(self.convs._modules['conv'],
                                 self.convs._modules['bn'])
            return convs
        else:
            return self.convs._modules['conv']

    def quant_convert(self):
        if not self.depthwise:
            if self.act_type == 'relu' and hasattr(self.convs, 'bn'):
                torch.quantization.fuse_modules(self.convs,
                                                ['conv', 'bn', 'act'],
                                                inplace=True)
            elif hasattr(self.convs, 'bn'):
                torch.quantization.fuse_modules(self.convs, ['conv', 'bn'],
                                                inplace=True)
            elif self.act_type == 'relu':
                torch.quantization.fuse_modules(self.convs, ['conv', 'act'],
                                                inplace=True)

            self.quant = True


class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self,
                 c1,
                 c2,
                 k=3,
                 s=1,
                 p=None,
                 g=1,
                 act=True,
                 deploy=False,
                 identity=False,
                 quant=False,
                 **args):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2
        self.quant = quant

        assert k == 3
        assert autopad(k, p) == 1

        if quant:
            self.quant_func = nn.quantized.FloatFunctional()
            identity = False

        self.act = nn.LeakyReLU(inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1,
                                         c2,
                                         k,
                                         s,
                                         autopad(k, p),
                                         groups=g,
                                         bias=True)

        else:
            self.rbr_identity = (nn.BatchNorm2d(
                num_features=c1) if c2 == c1 and s == 1 and identity else None)

            self.rbr_dense = Conv(c1, c2, k, s, groups=g, act_type=None)

            self.rbr_1x1 = Conv(c1, c2, 1, s, groups=g, act_type=None)

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0.
        else:
            id_out = self.rbr_identity(inputs)

        if self.quant:
            out = self.quant_func.add(self.rbr_dense(inputs),
                                      self.rbr_1x1(inputs))
            if self.rbr_identity is not None:
                out = self.quant_func.add(out, id_out)
            return self.act(out)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def fuse_params(self):
        if self.deploy:
            return

        self.rbr_dense = self.rbr_dense.get_fused_kernel()
        self.rbr_1x1 = self.rbr_1x1.get_fused_kernel()

        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight,
                                                      [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity,
                       (nn.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm))):

            identity_conv_1x1 = nn.Conv2d(in_channels=self.in_channels,
                                          out_channels=self.out_channels,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          groups=self.groups,
                                          bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(
                self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze(
            ).squeeze()

            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(
                2).unsqueeze(3)

            identity_conv_1x1 = fuse_conv_bn(identity_conv_1x1,
                                             self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(
                identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            bias_identity_expanded = torch.nn.Parameter(
                torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(
                torch.zeros_like(weight_1x1_expanded))

        self.rbr_dense.weight = torch.nn.Parameter(self.rbr_dense.weight +
                                                   weight_1x1_expanded +
                                                   weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias +
                                                 rbr_1x1_bias +
                                                 bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


class SEModule(nn.Module):
    """
    Apply SE on input, this won't change the shape of input.
    Each channel will have a unique weight.

    Params:
    channel -> int: input channel
    reduction -> int: the reduction factor for mid channels
    """
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=channel,
                               out_channels=channel // reduction,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=channel // reduction,
                               out_channels=channel,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        return x * identity


def drop_path(x,
              drop_prob: float = 0.,
              training: bool = False,
              scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    This op won't change the shape of input
    
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ResidualBlock(nn.Module):
    """
    basic residual block for CSP-Darknet
    """
    def __init__(self, in_ch, depthwise=False, quant=False, **args):
        super(ResidualBlock, self).__init__()
        self.quant = quant
        if quant:
            self.quant_func = nn.quantized.FloatFunctional()
        try:
            repconv = args['repconv']
        except:
            repconv = False

        self.conv1 = RepConv(in_ch, in_ch, identity=True, quant=quant, **
                             args) if repconv else Conv(
                                 in_ch, in_ch, k=3, p=1, depthwise=depthwise)
        self.conv2 = Conv(in_ch,
                          in_ch,
                          k=3,
                          p=1,
                          depthwise=depthwise,
                          act_type='lrelu',
                          bn=True)

    def forward(self, x):
        h = self.conv2(self.conv1(x))
        if self.quant:
            out = self.quant_func.add(x, h)
        else:
            out = x + h

        return out


class CSPStage(nn.Module):
    def __init__(self, c1, c2, n=1, depthwise=False, **args):
        super(CSPStage, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c1, c_, k=1)
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(in_ch=c_, depthwise=depthwise, **args)
            for i in range(n)
        ])
        self.cv3 = Conv(2 * c_, c2, k=1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.res_blocks(self.cv2(x))

        return self.cv3(torch.cat([y1, y2], dim=1))


class CustomELANBlock(nn.Module):
    """
    ELAN BLock of YOLOv7's backbone
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 model_size='tiny',
                 act_type='lrelu',
                 repconv=True,
                 depthwise=False,
                 **args):
        super(CustomELANBlock, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        if model_size == 'large':
            depth = 2
        elif model_size == 'tiny':
            depth = 1
        else:
            depth = 3
        self.cv1 = Conv(in_dim, inter_dim, k=3, p=1, act_type=act_type)
        self.cv2 = Conv(in_dim, inter_dim, k=3, p=1, act_type=act_type)
        self.cv3 = nn.Sequential(*[
            RepConv(inter_dim, inter_dim, identity=False
                    ) if repconv and i == 0 else Conv(inter_dim,
                                                      inter_dim,
                                                      k=3,
                                                      p=1,
                                                      act_type=None,
                                                      depthwise=depthwise,
                                                      **args)
            for i in range(depth)
        ])
        self.cv4 = nn.Sequential(*[
            RepConv(inter_dim, inter_dim, identity=False
                    ) if repconv and i == 0 else Conv(inter_dim,
                                                      inter_dim,
                                                      k=3,
                                                      p=1,
                                                      act_type=None,
                                                      depthwise=depthwise,
                                                      **args)
            for i in range(depth)
        ])

        self.out = Conv(inter_dim * 4, out_dim, k=1)

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, out_dim, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)

        # [B, C, H, W] -> [B, out_dim, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4], dim=1))
        return out


class CustomResidualELANBlock(nn.Module):
    """
    ELAN BLock of YOLOv7's backbone with skip connection
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 repconv=True,
                 model_size='tiny',
                 act_type='lrelu',
                 depthwise=False):
        super(CustomResidualELANBlock, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        if model_size == 'large':
            depth = 2
        elif model_size == 'tiny':
            depth = 1
        else:
            depth = 3
        self.cv1 = Conv(in_dim, inter_dim, k=3, p=1, act_type=act_type)
        self.cv2 = Conv(in_dim, inter_dim, k=3, p=1, act_type=act_type)
        self.cv3 = nn.Sequential(*[
            RepConv(inter_dim, inter_dim, identity=False
                    ) if repconv and i == 0 else Conv(inter_dim,
                                                      inter_dim,
                                                      k=3,
                                                      p=1,
                                                      act_type=None,
                                                      depthwise=depthwise)
            for i in range(depth)
        ])
        self.cv4 = nn.Sequential(*[
            RepConv(inter_dim, inter_dim, identity=False
                    ) if repconv and i == 0 else Conv(inter_dim,
                                                      inter_dim,
                                                      k=3,
                                                      p=1,
                                                      act_type=None,
                                                      depthwise=depthwise)
            for i in range(depth)
        ])
        if in_dim == out_dim:
            self.short_cut = nn.Sequential()
        else:
            self.short_cut = Conv(in_dim, out_dim, k=1, act_type=act_type)

        self.out = Conv(inter_dim * 4, out_dim, k=1)

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, out_dim, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)

        # [B, C, H, W] -> [B, out_dim, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4], dim=1))

        return self.short_cut(x) + out


class DownSample(nn.Module):
    def __init__(self, in_dim, act_type='lrelu'):
        super().__init__()
        inter_dim = in_dim // 2
        self.mp = nn.AvgPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type)
        self.cv2 = nn.Sequential(
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type))
        self.cv3 = Conv(in_dim, inter_dim, k=1, act_type=act_type)

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(self.cv3(x))

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out


class SkipBlock(nn.Module):
    """
    Skip Block: simple module designed to connect together the blocks with the different spatial sizes
    """
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 out_channels,
                 output_size,
                 kernel_size=3,
                 stride=1):
        super(SkipBlock, self).__init__()
        assert stride in [1, 2]
        self.output_size = output_size
        self.identity = stride == 1 and input_channels == out_channels

        self.core_block = nn.Sequential(
            # pw
            nn.Conv2d(input_channels, hidden_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(inplace=True),
            # dw
            nn.Conv2d(hidden_channels,
                      hidden_channels,
                      kernel_size,
                      stride, (kernel_size - 1) // 2,
                      groups=hidden_channels,
                      bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(inplace=True),

            # pw-linear
            nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, self.output_size)
        if self.identity:
            return x + self.core_block(x)
        else:
            return self.core_block(x)


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class Hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class ExpandedConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 expand_ratio=6,
                 skip_connection=False):
        super().__init__()

        self.stride = stride
        self.kernel_size = 3
        self.dilation = dilation
        self.expand_ratio = expand_ratio
        self.skip_connection = skip_connection
        middle_channels = in_channels * expand_ratio

        if self.expand_ratio != 1:
            # pointwise
            self.expand = nn.Sequential(
                OrderedDict([('conv',
                              nn.Conv2d(in_channels,
                                        middle_channels,
                                        1,
                                        bias=False)),
                             ('bn', nn.BatchNorm2d(middle_channels)),
                             ('relu', nn.ReLU6(inplace=True))]))

        # depthwise
        self.depthwise = nn.Sequential(
            OrderedDict([('conv',
                          nn.Conv2d(middle_channels,
                                    middle_channels,
                                    3,
                                    stride,
                                    dilation,
                                    dilation,
                                    groups=middle_channels,
                                    bias=False)),
                         ('bn', nn.BatchNorm2d(middle_channels)),
                         ('relu', nn.ReLU6(inplace=True))]))

        # project
        self.project = nn.Sequential(
            OrderedDict([('conv',
                          nn.Conv2d(middle_channels,
                                    out_channels,
                                    1,
                                    bias=False)),
                         ('bn', nn.BatchNorm2d(out_channels))]))

    def forward(self, x):
        if self.expand_ratio != 1:
            residual = self.project(self.depthwise(self.expand(x)))
        else:
            residual = self.project(self.depthwise(x))

        if self.skip_connection:
            outputs = x + residual
        else:
            outputs = residual
        return outputs


def depthwise_conv(in_channels,
                   out_channels,
                   kernel_size=3,
                   stride=1,
                   relu=False):
    return nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size,
                  stride,
                  kernel_size // 2,
                  groups=in_channels,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True) if relu else nn.Sequential(),
    )


class GhostConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 ratio=2,
                 stride=1,
                 dilation=1,
                 relu=True):
        super(GhostConv, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        padding = dilation if dilation > 1 else kernel_size // 2
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      init_channels,
                      kernel_size,
                      stride,
                      dilation=dilation * 1,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(init_channels),
            nn.LeakyReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels,
                      new_channels,
                      kernel_size,
                      1,
                      kernel_size // 2,
                      groups=init_channels,
                      bias=False),
            nn.BatchNorm2d(new_channels),
            nn.LeakyReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(3, 5)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, k=1, s=1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2=512, e=0.5, k=(3, 5)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.cv2 = Conv(c1, c_, k=1, s=1)
        self.cv3 = Conv(c_, c_, k=3, s=1)
        self.cv4 = Conv(c_, c_, k=1, s=1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv((1 + len(k)) * c_, c_, k=1, s=1)
        self.cv6 = Conv(c_, c_, k=3, s=1)
        self.cv7 = Conv(2 * c_, c2, k=1, s=1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class ELAN_W_Block(nn.Module):
    """
    ELAN-W BLock of YOLOv7's backbone
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 act_type='silu',
                 depthwise=False):
        super(ELAN_W_Block, self).__init__()
        inter_dim = int(in_dim * expand_ratio)

        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type)
        self.cv3 = Conv(inter_dim,
                        inter_dim,
                        k=3,
                        p=1,
                        act_type=act_type,
                        depthwise=depthwise)
        self.cv4 = Conv(inter_dim,
                        inter_dim,
                        k=3,
                        p=1,
                        act_type=act_type,
                        depthwise=depthwise)
        self.cv5 = Conv(inter_dim,
                        inter_dim,
                        k=3,
                        p=1,
                        act_type=act_type,
                        depthwise=depthwise)
        self.cv6 = Conv(inter_dim,
                        inter_dim,
                        k=3,
                        p=1,
                        act_type=act_type,
                        depthwise=depthwise)

        self.out = Conv(inter_dim * 6, out_dim, k=1)

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, 2C, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        x5 = self.cv5(x4)
        x6 = self.cv6(x5)

        # [B, C, H, W] -> [B, 2C, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4, x5, x6], dim=1))

        return out


class DownSample2(nn.Module):
    def __init__(self, in_dim, act_type='lrelu'):
        super().__init__()
        inter_dim = in_dim
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type))

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, 2C, H//2, W//2]
        """
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)

        # [B, 2C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out


class UpSample(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_dim, in_dim, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self,
                 scale_value=1.0,
                 bias_value=0.0,
                 scale_learnable=True,
                 bias_learnable=True,
                 inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """
    def __init__(self,
                 dim,
                 mlp_ratio=4,
                 out_features=None,
                 drop=0.,
                 bias=False,
                 **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)

        self.fc1 = nn.Conv2d(in_features,
                             hidden_features,
                             kernel_size=1,
                             padding=0,
                             stride=1,
                             bias=bias)
        self.act = nn.LeakyReLU(inplace=True)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features,
                             out_features,
                             kernel_size=1,
                             padding=0,
                             stride=1,
                             bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        '''
        BCHW -> BCHW
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim, 1, 1),
                                  requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self,
                 dim,
                 expansion_ratio=2,
                 act1_layer=StarReLU,
                 act2_layer=nn.Identity,
                 bias=False,
                 kernel_size=3,
                 padding=1,
                 **kwargs):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim,
                                 med_channels,
                                 kernel_size=1,
                                 padding=0,
                                 stride=1,
                                 bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(med_channels,
                                med_channels,
                                kernel_size=kernel_size,
                                padding=padding,
                                groups=med_channels,
                                bias=bias)  # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Conv2d(med_channels,
                                 dim,
                                 kernel_size=1,
                                 padding=0,
                                 stride=1,
                                 bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.dwconv(x)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self,
                 dim,
                 token_mixer=SepConv,
                 mlp=Mlp,
                 norm_layer=nn.BatchNorm2d,
                 drop=0.,
                 drop_path=0.,
                 layer_scale_init_value=1.,
                 res_scale_init_value=1.):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(
            dim=dim, init_value=layer_scale_init_value
        ) if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value
                                ) if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(
            dim=dim, init_value=layer_scale_init_value
        ) if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value
                                ) if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + self.layer_scale1(
            self.drop_path1(self.token_mixer(self.norm1(x))))
        x = self.res_scale2(x) + self.layer_scale2(
            self.drop_path2(self.mlp(self.norm2(x))))
        return x


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=16,
                        num_channels=in_channels,
                        eps=1e-6,
                        affine=True)


class AttnBlock(nn.Module):
    """
    Parameters:
        in_channels: int, input channels.
    Returns:
        out: attention output.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels,
                           in_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.k = nn.Conv2d(in_channels,
                           in_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.v = nn.Conv2d(in_channels,
                           in_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.proj_out = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3,
                                groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim,
            4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
