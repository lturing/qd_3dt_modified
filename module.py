import torch 
from torch import nn 
import torch.nn.functional as F
import numpy as np 
import math 
from functools import partial
from torchvision.ops import roi_align as tv_roi_align
from torch.nn.modules.utils import _pair
import torchvision

conv_cfg = {
    'Conv': nn.Conv2d,
    #'ConvWS': ConvWS2d,
    # TODO: octave conv
}

norm_cfg = {
    # format: layer_type: (abbreviation, module)
    'BN': ('bn', nn.BatchNorm2d),
    # 'SyncBN': ('bn', SyncBatchNorm2d),
    'GN': ('gn', nn.GroupNorm),
    # and potentially 'SN'
}


def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.

    Returns:
        layer (nn.Module): created conv layer
    """
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        # if layer_type == 'SyncBN':
        #     layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ConvModule(nn.Module):
    """Conv-Norm-Activation block.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        activation (str or None): Activation type, "ReLU" by default.
        inplace (bool): Whether to use inplace mode for activation.
        activate_last (bool): Whether to apply the activation layer in the
            last. (Do not use this flag since the behavior and api may be
            changed in the future.)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 activation='relu',
                 inplace=True,
                 activate_last=True):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.activate_last = activate_last

        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            print('ConvModule has norm and bias at the same time')

        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        if self.with_norm:
            norm_channels = out_channels if self.activate_last else in_channels
            self.norm_name, self.norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, self.norm)

        # build activation layer
        if self.with_activatation:
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported.'.format(
                    self.activation))
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)

        # Use msra init by default
        self.init_weights()


    def init_weights(self):
        nonlinearity = 'relu' if self.activation is None else self.activation
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        if self.activate_last:
            x = self.conv(x)
            if norm and self.with_norm:
                x = self.norm(x)
            if activate and self.with_activatation:
                x = self.activate(x)
        else:
            # WARN: this may be removed or modified
            if norm and self.with_norm:
                x = self.norm(x)
            if activate and self.with_activatation:
                x = self.activate(x)
            x = self.conv(x)
        return x


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(
            inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes)
        self.conv2 = nn.Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes)
        self.conv3 = nn.Conv2d(
            bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):

    def __init__(self,
                 levels,
                 block,
                 in_channels,
                 out_channels,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 dilation=1,
                 root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(
                in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(
                out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual)
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        self.stride = stride 
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children

        if self.stride > 1:
            row = x.shape[2]
            col = x.shape[3]
            padding_row = 0
            padding_col = 0
            if row % 2 == 1 or (row // 2 % 2 == 1):
                padding_row = self.stride // 2
            if col % 2 == 1 or (col // 2 % 2 == 1):
                padding_col = self.stride // 2
            
            downsample = nn.MaxPool2d(self.stride, stride=self.stride, padding=(padding_row, padding_col))

            bottom = downsample(x)
            #bottom = self.downsample(x) if self.downsample else x
        else:
            bottom = x 
            
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):

    def __init__(self,
                 levels,
                 channels,
                 num_classes=1000,
                 block_num=-1,
                 residual_root=False,
                 return_levels=False,
                 pool_size=7,
                 linear_root=False,
                 norm_eval=True):
        super(DLA, self).__init__()
        if block_num == 1:
            block = Bottleneck
        elif block_num == 2:
            block = BasicBlock
        self.norm_eval = norm_eval
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(
                3, channels[0], kernel_size=7, stride=1, padding=3,
                bias=False), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0],
                                            levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root)
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root)
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root)
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root)


    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(
                    inplanes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    padding=dilation,
                    bias=False,
                    dilation=dilation),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)
            ])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y[-4:]
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            return x

    def train(self, mode=True):
        super(DLA, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):

    def __init__(self,
                 node_kernel,
                 out_dim,
                 channels,
                 up_factors,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation='relu'):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = ConvModule(
                    c,
                    out_dim,
                    kernel_size=1,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=activation)

            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = nn.ConvTranspose2d(
                    out_dim,
                    out_dim,
                    f * 2,
                    stride=f,
                    padding=f // 2,
                    output_padding=0,
                    groups=out_dim,
                    bias=False)
                #fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            node = ConvModule(
                out_dim * 2,
                out_dim,
                kernel_size=node_kernel,
                stride=1,
                padding=node_kernel // 2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=activation)
            setattr(self, 'node_' + str(i), node)

        for m in self.modules():
            classname = m.__class__.__name__
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.channels) == len(layers), \
            '{} vs {} layers'.format(len(self.channels), len(layers))
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], 1))
            y.append(x)
        return x, y


class DLAUp(nn.Module):

    def __init__(self,
                 channels,
                 scales=(1, 2, 4, 8),
                 in_channels=None,
                 num_outs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation='relu'):
        super(DLAUp, self).__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        self.last_conv = ConvModule(
            in_channels[-1],
            channels[-1],
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=activation)
        self.num_outs = num_outs
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self, 'ida_{}'.format(i),
                IDAUp(3, channels[j], in_channels[j:],
                      scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1

        ms_feat = [layers[-1]]
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])  # y : aggregation nodes
            layers[-i - 1:] = y
            ms_feat.append(x)
        ms_feat = ms_feat[::-1]
        ms_feat[-1] = self.last_conv(ms_feat[-1])
        if self.num_outs > len(ms_feat):
            ms_feat.append(F.max_pool2d(ms_feat[-1], 1, stride=2))
        return ms_feat  # x


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class AnchorGenerator(object):

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)

        base_anchors = torch.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            dim=-1).round()

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cpu'):
        base_anchors = self.base_anchors.to(device)

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cpu'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(
            valid.size(0), self.num_base_anchors).contiguous().view(-1)
        return valid


def delta2bbox(rois,
               deltas,
               means=[0, 0, 0, 0],
               stds=[1, 1, 1, 1],
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    #gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    #gx = torch.addcmul(px, pw, dx)
    gx = px + pw * dx 

    #gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    #gy = torch.addcmul(py, ph, dy)
    gy = py + ph * dy

    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes

def batched_nms(bboxes, scores, inds, nms_cfg):
    """Performs non-maximum suppression in a batched fashion.
    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.
    Arguments:
        bboxes (torch.Tensor): bboxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        inds (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different inds,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
    Returns:
        tuple: kept bboxes and indice.
    """
    max_coordinate = bboxes.max()
    offsets = inds.to(bboxes) * (max_coordinate + 1)
    bboxes_for_nms = bboxes + offsets[:, None]
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)
    #keep = torchvision.ops.nms(crop_bboxes, crop_scores, self.args.iou)  # NMS
    dets, keep = nms_op(
        torch.cat([bboxes_for_nms, scores[:, None]], -1), **nms_cfg_)
    bboxes = bboxes[keep]
    scores = dets[:, -1]
    return torch.cat([bboxes, scores[:, None]], -1), keep


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    scores = multi_scores[:, 1:]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        return bboxes, labels


    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
    return dets[:max_num], labels[keep[:max_num]]


def multiclass_3d_nms(multi_bboxes,
                      multi_scores,
                      depth_pred,
                      depth_uncertainty_pred,
                      dim_pred,
                      rot_pred,
                      cen_2d_pred,
                      score_thr,
                      nms_cfg,
                      max_num=-1,
                      score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        depth_pred (Tensor): shape (n, 1), estimated depth.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """

    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        reg_class_agnostic = False
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)[:, 1:]
    else:
        reg_class_agnostic = True
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    scores = multi_scores[:, 1:]

    # filter out boxes with low scores
    if depth_uncertainty_pred is not None:
        valid_mask = (scores * depth_uncertainty_pred[:, 1:]) > score_thr
    else:
        valid_mask = scores > score_thr
    
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]

    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        depths = multi_bboxes.new_zeros((0, ))
        depths_uncertainty = multi_bboxes.new_zeros((0, ))
        dims = multi_bboxes.new_zeros((0, 3))
        rots = multi_bboxes.new_zeros((0, ))
        cen_2ds = multi_bboxes.new_zeros((0, 2))
        return bboxes, labels, depths, depths_uncertainty, dims, rots, cen_2ds

    if 1:
        nms_cfg['iou_thr'] = 0.4
        dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
    else:
        #keep = torchvision.ops.nms(crop_bboxes, crop_scores, self.args.iou)  # NMS
        keep = torchvision.ops.nms(bboxes, scores, 0.15) #nms_cfg["iou_thr"])  # NMS
        dets = torch.cat([bboxes[keep], scores[keep].unsqueeze(1)], dim=1)

    if depth_pred is not None:
        if reg_class_agnostic:
            depths = depth_pred.expand(-1, num_classes, -1)[valid_mask]
        else:
            depths = depth_pred.view(multi_scores.size(0), -1,
                                     1)[:, 1:][valid_mask]
        depths = depths[keep[:max_num]]
    else:
        depths = None

    if depth_uncertainty_pred is not None:
        if reg_class_agnostic:
            depths_uncertainty = depth_uncertainty_pred.expand(
                -1, num_classes, -1)[valid_mask]
        else:
            depths_uncertainty = depth_uncertainty_pred.view(
                multi_scores.size(0), -1, 1)[:, 1:][valid_mask]
        depths_uncertainty = depths_uncertainty[keep[:max_num]]
    else:
        depths_uncertainty = None

    if dim_pred is not None:
        if reg_class_agnostic:
            dims = dim_pred.expand(-1, num_classes, -1)[valid_mask]
        else:
            dims = dim_pred.view(multi_scores.size(0), -1, 3)[:,
                                                              1:][valid_mask]
        dims = dims[keep[:max_num]]
    else:
        dims = None

    if rot_pred is not None:
        if reg_class_agnostic:
            rots = rot_pred.expand(-1, num_classes, -1)[valid_mask]
        else:
            rots = rot_pred.view(multi_scores.size(0), -1, 1)[:,
                                                              1:][valid_mask]
        rots = rots[keep[:max_num]]
    else:
        rots = None

    if cen_2d_pred is not None:
        if reg_class_agnostic:
            cen_2ds = cen_2d_pred.expand(-1, num_classes, -1)[valid_mask]
        else:
            cen_2ds = cen_2d_pred.view(multi_scores.size(0), -1,
                                       2)[:, 1:][valid_mask]
        cen_2ds = cen_2ds[keep[:max_num]]
    else:
        cen_2ds = None

    dets = dets[:max_num]
    labels = labels[keep[:max_num]]
    scores = dets[:, -1].reshape(-1)
    valid_index = scores > 0.6

    data = [dets, labels, depths, depths_uncertainty, dims, rots, cen_2ds]
    for i in range(len(data)):
        if data[i] is not None:
            data[i] = data[i][valid_index]
    
    return data 
    return dets, labels, depths, depths_uncertainty, dims, rots, cen_2ds



class AnchorHead(nn.Module):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_scales=[8, 16, 32],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)):
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        self._init_layers()

    def _init_layers(self):
        self.conv_cls = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list


    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_scores, bbox_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg['score_thr'], cfg['nms'],
                                                cfg['max_per_img'])

        return det_bboxes, det_labels



def nms_cpu(dets, thresh):
    dets = dets.numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.
    """
    ''' 
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu'
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))
    ''' 

    ''' 
    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        assert not dets_th.is_cuda
        inds = nms_cpu(dets_th, iou_thr)
        
        if dets_th.is_cuda:
            inds = nms_cuda.nms(dets_th, iou_thr)
        else:
            inds = nms_cpu.nms(dets_th, iou_thr)
        
    '''
    dets_th = dets
    inds = nms_cpu(dets_th, iou_thr)

    return dets[inds, :], inds



class RPNHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(RPNHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred


    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                _, topk_inds = scores.topk(nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
                
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)

            if cfg['min_bbox_size'] > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg['min_bbox_size']) &
                                           (h >= cfg['min_bbox_size'])).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg['nms_thr'])
            proposals = proposals[:cfg['nms_post'], :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)

        if cfg['nms_across_levels']:
            proposals, _ = nms(proposals, cfg['nms_thr'])
            proposals = proposals[:cfg['max_num'], :]
        else:
            scores = proposals[:, 4]
            num = min(cfg['max_num'], proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        
        return proposals



class RoIAlign(nn.Module):

    def __init__(self,
                 out_size,
                 spatial_scale,
                 sample_num=0,
                 use_torchvision=True):
        super(RoIAlign, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.use_torchvision = use_torchvision

    def forward(self, features, rois):
        return tv_roi_align(features, rois, _pair(self.out_size),
                                self.spatial_scale, self.sample_num)



def bbox2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    assert proposals.size() == gt.size()
    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def bbox_target_single(pos_bboxes,
                       neg_bboxes,
                       pos_gt_bboxes,
                       pos_gt_labels,
                       pos_gt_depths,
                       pos_gt_alphas,
                       pos_gt_rotys,
                       pos_gt_dims,
                       pos_gt_2dc,
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
    depth_targets = pos_bboxes.new_zeros(num_samples, )
    depth_weights = pos_bboxes.new_zeros(num_samples, )
    alpha_targets = pos_bboxes.new_zeros(num_samples, )
    alpha_weights = pos_bboxes.new_zeros(num_samples, )
    roty_targets = pos_bboxes.new_zeros(num_samples, )
    roty_weights = pos_bboxes.new_zeros(num_samples, )
    dim_targets = pos_bboxes.new_zeros(num_samples, 3)
    dim_weights = pos_bboxes.new_zeros(num_samples, 3)
    cen_2d_targets = pos_bboxes.new_zeros(num_samples, 2)
    cen_2d_weights = pos_bboxes.new_zeros(num_samples, 2)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg['pos_weight'] <= 0 else cfg['pos_weight']
        label_weights[:num_pos] = pos_weight

        pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means,
                                      target_stds)
        if pos_bbox_targets is not None:
            bbox_targets[:num_pos, :] = pos_bbox_targets[:num_pos]
            bbox_weights[:num_pos, :] = 1.0

        if pos_gt_depths is not None:
            depth_targets[:num_pos] = pos_gt_depths[:num_pos]
            depth_weights[:num_pos] = 1.0

        if pos_gt_alphas is not None:
            alpha_targets[:num_pos] = pos_gt_alphas[:num_pos]
            alpha_weights[:num_pos] = 1.0

        if pos_gt_rotys is not None:
            roty_targets[:num_pos] = pos_gt_rotys[:num_pos]
            roty_weights[:num_pos] = 1.0

        if pos_gt_dims is not None:
            dim_targets[:num_pos, :] = pos_gt_dims[:num_pos]
            dim_weights[:num_pos, :] = 1.0

        if pos_gt_2dc is not None:
            cen_2d_targets[:num_pos, :] = pos_gt_2dc[:num_pos]
            cen_2d_weights[:num_pos, :] = 1.0
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0

    return labels, label_weights, bbox_targets, bbox_weights, \
           depth_targets, depth_weights, alpha_targets, alpha_weights, \
           roty_targets, roty_weights, dim_targets, dim_weights, \
           cen_2d_targets, cen_2d_weights


def bbox_target(pos_bboxes_list,
                neg_bboxes_list,
                pos_gt_bboxes_list,
                pos_gt_labels_list,
                cfg,
                reg_classes=1,
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
                concat=True):
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        bbox_target_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_bboxes_list,
        pos_gt_labels_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)

    return labels, label_weights, bbox_targets, bbox_weights


def bbox_3d_target(pos_bboxes_list,
                   neg_bboxes_list,
                   pos_gt_bboxes_list,
                   pos_gt_labels_list,
                   pos_gt_depth_list,
                   pos_gt_alpha_list,
                   pos_gt_roty_list,
                   pos_gt_dim_list,
                   pos_gt_2dc_list,
                   cfg,
                   reg_classes=1,
                   target_means=[.0, .0, .0, .0],
                   target_stds=[1.0, 1.0, 1.0, 1.0],
                   concat=True):
    labels, label_weights, \
    bbox_targets, bbox_weights, \
    depth_targets, depth_weights, \
    alpha_targets, alpha_weights, \
    roty_targets, roty_weights, \
    dim_targets, dim_weights, \
    cen_2d_targets, cen_2d_weights = multi_apply(
        bbox_target_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_bboxes_list,
        pos_gt_labels_list,
        pos_gt_depth_list,
        pos_gt_alpha_list,
        pos_gt_roty_list,
        pos_gt_dim_list,
        pos_gt_2dc_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
        depth_targets = torch.cat(depth_targets, 0)
        depth_weights = torch.cat(depth_weights, 0)
        alpha_targets = torch.cat(alpha_targets, 0)
        alpha_weights = torch.cat(alpha_weights, 0)
        roty_targets = torch.cat(roty_targets, 0)
        roty_weights = torch.cat(roty_weights, 0)
        dim_targets = torch.cat(dim_targets, 0)
        dim_weights = torch.cat(dim_weights, 0)
        cen_2d_targets = torch.cat(cen_2d_targets, 0)
        cen_2d_weights = torch.cat(cen_2d_weights, 0)

    return labels, label_weights, bbox_targets, bbox_weights, \
           depth_targets, depth_weights, alpha_targets, alpha_weights, \
           roty_targets, roty_weights, dim_targets, dim_weights, \
           cen_2d_targets, cen_2d_weights


class BBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=81,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(BBoxHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic
        self.fp16_enabled = False
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            in_channels *= (self.roi_feat_size * self.roi_feat_size)
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, self.cls_out_channels)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else (4 *
                                                        self.cls_out_channels)
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

    def init_weights(self):
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def get_target(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes=reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            bboxes /= scale_factor

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg['score_thr'], cfg['nms'],
                                                    cfg['max_per_img'])
            return det_bboxes, det_labels

    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds])

        return bboxes_list

    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = delta2bbox(rois, bbox_pred, self.target_means,
                                  self.target_stds, img_meta['img_shape'])
        else:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois


class ConvFCBBoxHead(BBoxHead):
    """More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= (self.roi_feat_size * self.roi_feat_size)

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.cls_out_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.cls_out_channels)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def get_embeds(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        return x_cls, x_reg

    def get_logits(self, x_cls, x_reg):
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

    def forward(self, x):
        x_cls, x_reg = self.get_embeds(x)
        cls_score, bbox_pred = self.get_logits(x_cls, x_reg)

        return cls_score, bbox_pred


class ConvFCBBox3DRotSepConfidenceHead(nn.Module):
    """More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_dep_convs=0,
                 num_dep_fcs=0,
                 num_dim_convs=0,
                 num_dim_fcs=0,
                 num_rot_convs=0,
                 num_rot_fcs=0,
                 num_2dc_convs=0,
                 num_2dc_fcs=0,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 num_classes=8,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 center_scale=10,
                 reg_class_agnostic=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_depth=None,
                 with_depth=False,
                 loss_uncertainty=None,
                 with_uncertainty=False,
                 use_uncertainty=False,
                 loss_dim=None,
                 with_dim=False,
                 loss_rot=None,
                 with_rot=False,
                 loss_2dc=None,
                 with_2dc=False,
                 loss_cls=None,
                 loss_bbox=None,
                 *args,
                 **kwargs):
        super(ConvFCBBox3DRotSepConfidenceHead, self).__init__()
        assert (num_shared_convs + num_shared_fcs + num_dep_convs +
                num_dep_fcs + num_dim_convs + num_dim_fcs + num_rot_convs +
                num_rot_fcs + num_2dc_convs + num_2dc_fcs > 0)
        if num_dep_convs > 0 or num_dim_convs > 0 \
                or num_rot_convs or num_2dc_convs:
            assert num_shared_fcs == 0
        if not with_depth:
            assert num_dep_convs == 0 and num_dep_fcs == 0
        if not with_dim:
            assert num_dim_convs == 0 and num_dim_fcs == 0
        if not with_rot:
            assert num_rot_convs == 0 and num_rot_fcs == 0
        if not with_2dc:
            assert num_2dc_convs == 0 and num_2dc_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_dep_convs = num_dep_convs
        self.num_dep_fcs = num_dep_fcs
        self.num_dim_convs = num_dim_convs
        self.num_dim_fcs = num_dim_fcs
        self.num_rot_convs = num_rot_convs
        self.num_rot_fcs = num_rot_fcs
        self.num_2dc_convs = num_2dc_convs
        self.num_2dc_fcs = num_2dc_fcs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.reg_class_agnostic = reg_class_agnostic
        self.target_means = target_means
        self.target_stds = target_stds
        self.center_scale = center_scale
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.with_depth = with_depth

        self.with_uncertainty = with_uncertainty
        self.use_uncertainty = use_uncertainty
        self.with_dim = with_dim
        self.with_rot = with_rot
        self.with_2dc = with_2dc
        self.cls_out_channels = num_classes
        self.num_classes = num_classes

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add depth specific branch
        self.dep_convs, self.dep_fcs, self.dep_last_dim = \
            self._add_conv_fc_branch(
                self.num_dep_convs, self.num_dep_fcs, self.shared_out_channels)

        # add dim specific branch
        self.dim_convs, self.dim_fcs, self.dim_last_dim = \
            self._add_conv_fc_branch(
                self.num_dim_convs, self.num_dim_fcs, self.shared_out_channels)

        # add rot specific branch
        self.rot_convs, self.rot_fcs, self.rot_last_dim = \
            self._add_conv_fc_branch(
                self.num_rot_convs, self.num_rot_fcs,
                self.shared_out_channels)

        # add 2dc specific branch
        self.cen_2d_convs, self.cen_2d_fcs, self.cen_2d_last_dim = \
            self._add_conv_fc_branch(
                self.num_2dc_convs, self.num_2dc_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0:
            if self.num_dep_fcs == 0:
                self.dep_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_dim_fcs == 0:
                self.dim_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_rot_fcs == 0:
                self.rot_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_2dc_fcs == 0:
                self.cen_2d_last_dim *= (
                    self.roi_feat_size * self.roi_feat_size)

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_depth:
            out_dim_dep = (1 if self.reg_class_agnostic else
                           self.cls_out_channels)
            if self.with_uncertainty:
                self.fc_dep_uncer = nn.Linear(self.dep_last_dim, out_dim_dep)
            self.fc_dep = nn.Linear(self.dep_last_dim, out_dim_dep)
        if self.with_dim:
            out_dim_size = (3 if self.reg_class_agnostic else 3 *
                            self.cls_out_channels)
            self.fc_dim = nn.Linear(self.dim_last_dim, out_dim_size)
        if self.with_rot:
            out_rot_size = (8 if self.reg_class_agnostic else 8 *
                            self.cls_out_channels)
            self.fc_rot = nn.Linear(self.rot_last_dim, out_rot_size)
        if self.with_2dc:
            out_2dc_size = (2 if self.reg_class_agnostic else 2 *
                            self.cls_out_channels)
            self.fc_2dc = nn.Linear(self.cen_2d_last_dim, out_2dc_size)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared or self.num_shared_fcs == 0):
                last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        module_lists = [self.shared_fcs]
        if self.with_depth:
            if self.with_uncertainty:
                module_lists += [self.fc_dep_uncer]
            module_lists += [self.fc_dep, self.dep_fcs]
        if self.with_dim:
            module_lists += [self.fc_dim, self.dim_fcs]
        if self.with_rot:
            module_lists += [self.fc_rot, self.rot_fcs]
        if self.with_2dc:
            module_lists += [self.fc_2dc, self.cen_2d_fcs]

        for module_list in module_lists:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def get_embeds(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # separate branches
        x_dep = x
        x_dim = x
        x_rot = x
        x_2dc = x

        for conv in self.dep_convs:
            x_dep = conv(x_dep)
        if x_dep.dim() > 2:
            x_dep = x_dep.view(x_dep.size(0), -1)
        for fc in self.dep_fcs:
            x_dep = self.relu(fc(x_dep))

        for conv in self.dim_convs:
            x_dim = conv(x_dim)
        if x_dim.dim() > 2:
            x_dim = x_dim.view(x_dim.size(0), -1)
        for fc in self.dim_fcs:
            x_dim = self.relu(fc(x_dim))

        for conv in self.rot_convs:
            x_rot = conv(x_rot)
        if x_rot.dim() > 2:
            x_rot = x_rot.view(x_rot.size(0), -1)
        for fc in self.rot_fcs:
            x_rot = self.relu(fc(x_rot))

        for conv in self.cen_2d_convs:
            x_2dc = conv(x_2dc)
        if x_2dc.dim() > 2:
            x_2dc = x_2dc.view(x_2dc.size(0), -1)
        for fc in self.cen_2d_fcs:
            x_2dc = self.relu(fc(x_2dc))

        return x_dep, x_dim, x_rot, x_2dc

    def get_logits(self, x_dep, x_dim, x_rot, x_2dc):

        def get_rot(pred):
            pred = pred.view(pred.size(0), -1, 8)

            # bin 1
            divider1 = torch.sqrt(pred[:, :, 2:3]**2 + pred[:, :, 3:4]**2 +
                                  1e-10)
            b1sin = pred[:, :, 2:3] / divider1
            b1cos = pred[:, :, 3:4] / divider1

            # bin 2
            divider2 = torch.sqrt(pred[:, :, 6:7]**2 + pred[:, :, 7:8]**2 +
                                  1e-10)
            b2sin = pred[:, :, 6:7] / divider2
            b2cos = pred[:, :, 7:8] / divider2

            rot = torch.cat(
                [pred[:, :, 0:2], b1sin, b1cos, pred[:, :, 4:6], b2sin, b2cos],
                2)
            return rot

        depth_pred = self.fc_dep(x_dep) if self.with_depth else None
        depth_uncertainty_pred = self.fc_dep_uncer(
            x_dep) if self.with_depth and self.with_uncertainty else None
        dim_pred = self.fc_dim(x_dim) if self.with_dim else None
        rot_pred = get_rot(self.fc_rot(x_rot)) if self.with_rot else None
        cen_2d_pred = self.fc_2dc(x_2dc) if self.with_2dc else None

        return depth_pred, depth_uncertainty_pred, dim_pred, rot_pred, cen_2d_pred

    def forward(self, x):
        x_dep, x_dim, x_rot, x_2dc = self.get_embeds(x)
        depth_pred, depth_uncertainty_pred, dim_pred, rot_pred, cen_2d_pred = \
            self.get_logits(x_dep, x_dim, x_rot, x_2dc)

        return depth_pred, depth_uncertainty_pred, dim_pred, rot_pred, cen_2d_pred

    def get_target(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        pos_gt_depths = [res.pos_gt_depths for res in sampling_results]
        pos_gt_alphas = [res.pos_gt_alphas for res in sampling_results]
        pos_gt_rotys = [res.pos_gt_rotys for res in sampling_results]
        pos_gt_dims = [res.pos_gt_dims for res in sampling_results]
        pos_gt_2dcs = [res.pos_gt_2dcs for res in sampling_results]

        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_3d_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            pos_gt_depths,
            pos_gt_alphas,
            pos_gt_rotys,
            pos_gt_dims,
            pos_gt_2dcs,
            rcnn_train_cfg,
            reg_classes=reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets


    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       depth_pred,
                       depth_uncertainty_pred,
                       dim_pred,
                       rot_pred,
                       cen_2d_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            bboxes /= scale_factor

        def get_depth(pred, scale: float = 2.0):
            return torch.exp(pred / scale).view(pred.size(0), -1, 1)

        def get_uncertainty_prob(depth_uncertainty_pred):
            if depth_uncertainty_pred is None:
                return None
            return torch.clamp(
                depth_uncertainty_pred, min=0.0, max=1.0)

        def get_dim(pred, scale: float = 2.0):
            return torch.exp(pred / scale).view(pred.size(0), -1, 3)

        def get_alpha(rot):
            """Generate alpha value from predicted CLSREG array

            Args:
                rot (torch.Tensor): rotation CLSREG array: (B, num_classes, 8) 
                [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]

            Returns:
                torch.Tensor: (B, num_classes, 1)
            """
            alpha1 = torch.atan(
                rot[:, :, 2:3] / rot[:, :, 3:4]) + (-0.5 * np.pi)
            alpha2 = torch.atan(
                rot[:, :, 6:7] / rot[:, :, 7:8]) + (0.5 * np.pi)
            # Model is not decisive at index overlap region
            # Could be unstable for the alpha estimation
            # idx1 = (rot[:, :, 1:2] > rot[:, :, 0:1]).float()
            # idx2 = (rot[:, :, 5:6] > rot[:, :, 4:5]).float()
            # alpha = (alpha1 * idx1 + alpha2 * idx2)
            # alpha /= (idx1 + idx2 + ((idx1 + idx2) == 0))
            idx = (rot[:, :, 1:2] > rot[:, :, 5:6]).float()
            alpha = alpha1 * idx + alpha2 * (1 - idx)
            return alpha

        def get_delta_2d(delta_cen, scale: float = 10.0):
            return delta_cen.view(delta_cen.size(0), -1, 2) * scale

        def get_box_cen(bbox):
            return torch.cat(
                [(bbox[:, 0::4, None] + bbox[:, 2::4, None]) / 2.0,
                 (bbox[:, 1::4, None] + bbox[:, 3::4, None]) / 2.0],
                dim=2)

        def get_cen2d(delta_2d, box_cen):
            return delta_2d + box_cen

        depth_pred = get_depth(depth_pred)
        depth_uncertainty_prob = get_uncertainty_prob(depth_uncertainty_pred)
        if not self.use_uncertainty:
            depth_uncertainty_prob = depth_uncertainty_prob * 0. + 1.0
        dim_pred = get_dim(dim_pred)
        rot_pred = get_alpha(rot_pred)
        delta_2d = get_delta_2d(cen_2d_pred, scale=self.center_scale)
        cen2d_pred = get_cen2d(delta_2d, get_box_cen(bboxes.detach()))

        if cfg is None:
            return bboxes, scores, depth_pred, depth_uncertainty_prob, dim_pred, rot_pred, cen2d_pred
        else:
            det_bboxes, det_labels, det_depths, det_depth_uncertainty, dim_preds, rot_preds, cen2d_preds = \
                multiclass_3d_nms(
                    bboxes,
                    scores,
                    depth_pred,
                    depth_uncertainty_prob,
                    dim_pred,
                    rot_pred,
                    cen2d_pred,
                    cfg['score_thr'], cfg['nms'],
                    cfg['max_per_img'])
            return det_bboxes, det_labels, det_depths, det_depth_uncertainty, dim_preds, rot_preds, cen2d_preds


class SingleRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56):
        super(SingleRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.fp16_enabled = False

    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        roi_layers = nn.ModuleList(
            [RoIAlign(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def forward(self, feats, rois):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        roi_feats = feats[0].new_zeros(rois.size()[0], self.out_channels,
                                       out_size, out_size)

        for i in range(num_levels):
            roi_feats += feats[i].max() * 0
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] += roi_feats_t

        return roi_feats


def get_similarity(key_embeds, ref_embeds, tau=-1, norm=False):
    pre_norm = True if tau > 0 else False
    if norm or pre_norm:
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)
    sim = torch.mm(key_embeds, ref_embeds.t())
    if tau > 0:
        sim /= tau
    return sim


class MultiPos3DTrackHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 num_fcs=1,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 embed_channels=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_depth=None,
                 loss_asso=None,
                 loss_iou=None):
        super(MultiPos3DTrackHead, self).__init__()
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.embed_channels = embed_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.relu = nn.ReLU(inplace=True)
        self.convs, self.fcs, last_layer_dim = self._add_conv_fc_branch(
            self.num_convs, self.num_fcs, self.in_channels)
        self.fc_embed = nn.Linear(last_layer_dim, embed_channels)
        self.depth_embed = nn.Linear(embed_channels, 1)
        self.loss_asso_tau = loss_asso.pop('tau', -1)
        self.loss_asso = loss_asso
        self.loss_depth = loss_depth
        self.loss_iou = loss_iou

    def _add_conv_fc_branch(self, num_convs, num_fcs, in_channels):
        last_layer_dim = in_channels
        # add branch specific conv layers
        convs = nn.ModuleList()
        if num_convs > 0:
            for i in range(num_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        fcs = nn.ModuleList()
        if num_fcs > 0:
            last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return convs, fcs, last_layer_dim

    def init_weights(self):
        for m in self.fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.fc_embed.weight, 0, 0.01)
        nn.init.constant_(self.fc_embed.bias, 0)
        nn.init.normal_(self.depth_embed.weight, 0, 0.01)
        nn.init.constant_(self.depth_embed.bias, 0)

    def forward(self, x):
        if self.num_convs > 0:
            for i, conv in enumerate(self.convs):
                x = conv(x)
        x = x.view(x.size(0), -1)
        if self.num_fcs > 0:
            for i, fc in enumerate(self.fcs):
                x = self.relu(fc(x))
        x = self.fc_embed(x)
        depth = self.depth_embed(x)
        return x, depth

    def match(self,
              key_embeds=None,
              ref_gt_embeds=None,
              ref_pos_embeds=None,
              ref_neg_embeds=None,
              key_neg_embeds=None,
              key_sampling_results=None,
              ref_sampling_results=None,
              img_meta=None,
              cfg=None):
        matrix = []
        cos_matrix = []
        n = len(key_sampling_results)
        # get key embeds
        if cfg['with_key_pos']:
            num_keys = [res.pos_bboxes.size(0) for res in key_sampling_results]
        else:
            num_keys = [res.num_gts for res in key_sampling_results]
        key_embeds = torch.split(key_embeds, num_keys)

        # get ref gt embeds
        num_ref_gts = [
            res.ref_gt_bboxes.size(0) for res in key_sampling_results
        ]
        ref_gt_embeds = torch.split(ref_gt_embeds, num_ref_gts)

        if cfg['with_ref_pos']:
            num_ref_pos = [
                res.pos_bboxes.size(0) for res in ref_sampling_results
            ]
            ref_pos_embeds = torch.split(ref_pos_embeds, num_ref_pos)

        # get ref neg embeds
        if cfg['with_ref_neg']:
            num_ref_negs = [
                res.neg_bboxes.size(0) for res in ref_sampling_results
            ]
            ref_neg_embeds = torch.split(ref_neg_embeds, num_ref_negs)

        if cfg['with_key_neg']:
            num_key_negs = [
                res.neg_bboxes.size(0) for res in key_sampling_results
            ]
            key_neg_embeds = torch.split(key_neg_embeds, num_key_negs)

        for i in range(n):
            ref_embeds = [ref_gt_embeds[i]]
            if cfg['with_ref_pos']:
                ref_embeds.append(ref_pos_embeds[i])
            if cfg['with_ref_neg']:
                ref_embeds.append(ref_neg_embeds[i])
            if cfg['with_key_neg']:
                ref_embeds.append(key_neg_embeds[i])
            ref_embeds = torch.cat(ref_embeds, dim=0)
            _matrix = get_similarity(
                key_embeds[i], ref_embeds, tau=self.loss_asso_tau)
            matrix.append(_matrix)
            _cos_matrix = get_similarity(
                key_embeds[i], ref_embeds, norm=True, tau=-1)
            cos_matrix.append(_cos_matrix)

        return matrix, cos_matrix

    def get_asso_targets(self, sampling_results, gt_pids, cfg):
        ids = []
        id_weights = []
        for i, res in enumerate(sampling_results):
            if cfg['with_key_pos']:
                _ids = res.pos_gt_pids
            else:
                _ids = gt_pids[i]
            _id_weights = torch.ones_like(_ids, dtype=torch.float)
            num_ref_gts = res.ref_gt_bboxes.size(0)
            gt_is_dummy = torch.nonzero(_ids == -1).squeeze()
            _ids[gt_is_dummy] = 0
            _id_weights[gt_is_dummy] = 0.
            ids.append(_ids)
            id_weights.append(_id_weights)
        return ids, id_weights

