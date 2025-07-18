import math

import numpy
import torch


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, k=3, p=1),
                                         Conv(ch, ch, k=3, p=1))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch)
        self.conv2 = Conv((2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(tensors=y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat(tensors=[x, y1, y2, self.res_m(y2)], dim=1))


class Backbone(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(width[0], width[1], k=3, s=2, p=1))
        # p2/4
        self.p2.append(Conv(width[1], width[2], k=3, s=2, p=1))
        self.p2.append(CSP(width[2], width[2], depth[0]))
        # p3/8
        self.p3.append(Conv(width[2], width[3], k=3, s=2, p=1))
        self.p3.append(CSP(width[3], width[3], depth[1]))
        # p4/16
        self.p4.append(Conv(width[3], width[4], k=3, s=2, p=1))
        self.p4.append(CSP(width[4], width[4], depth[2]))
        # p5/32
        self.p5.append(Conv(width[4], width[5], k=3, s=2, p=1))
        self.p5.append(CSP(width[5], width[5], depth[0]))
        self.p5.append(SPP(width[5], width[5]))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class Neck(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], add=False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], add=False)
        self.h3 = Conv(width[3], width[3], k=3, s=2, p=1)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], add=False)
        self.h5 = Conv(width[4], width[4], k=3, s=2, p=1)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], add=False)

    def forward(self, x):
        p3, p4, p5 = x
        p4 = self.h1(torch.cat(tensors=[self.up(p5), p4], dim=1))
        p3 = self.h2(torch.cat(tensors=[self.up(p4), p3], dim=1))
        p4 = self.h4(torch.cat(tensors=[self.h3(p3), p4], dim=1))
        p5 = self.h6(torch.cat(tensors=[self.h5(p4), p5], dim=1))
        return p3, p4, p5


class Head(torch.nn.Module):
    def __init__(self, filters, nc=80):
        super().__init__()
        self.nc = nc
        self.nl = len(filters)  # number of detection layers
        self.stride = torch.zeros(self.nl)  # strides computed during build

        self.m = torch.nn.ModuleList()
        self.cls = torch.nn.ModuleList()
        self.box = torch.nn.ModuleList()
        self.obj = torch.nn.ModuleList()

        for i in range(len(filters)):
            self.m.append(torch.nn.Sequential(Conv(filters[i], filters[i], k=3, p=1),
                                              Conv(filters[i], filters[i], k=3, p=1)))

            self.box.append(torch.nn.Conv2d(filters[i], out_channels=4, kernel_size=1))
            self.obj.append(torch.nn.Conv2d(filters[i], out_channels=1, kernel_size=1))
            self.cls.append(torch.nn.Conv2d(filters[i], out_channels=self.nc, kernel_size=1))
        for box, cls, obj in zip(self.box, self.cls, self.obj):
            box.bias.data.fill_(1.0)
            cls.bias.data.fill_(float(-numpy.log((1 - 0.01) / 0.01)))
            obj.bias.data.fill_(float(-numpy.log((1 - 0.01) / 0.01)))

    def forward(self, x):
        x = [m(i) for i, m in zip(x, self.m)]

        cls = [m(i) for i, m in zip(x, self.cls)]
        box = [m(i) for i, m in zip(x, self.box)]
        obj = [m(i) for i, m in zip(x, self.obj)]

        if self.training:
            return cls, box, obj

        n = cls[0].shape[0]
        sizes = [i.shape[2:] for i in cls]
        anchors = self.__make_anchors(sizes, self.stride, cls[0].device, cls[0].dtype)

        cls = [i.permute(0, 2, 3, 1).reshape(n, -1, self.nc) for i in cls]
        box = [i.permute(0, 2, 3, 1).reshape(n, -1, 4) for i in box]
        obj = [i.permute(0, 2, 3, 1).reshape(n, -1) for i in obj]

        box = torch.cat(box, dim=1)
        obj = torch.cat(obj, dim=1).sigmoid()
        cls = torch.cat(cls, dim=1).sigmoid()

        box = self.__box_decode(torch.cat(anchors), box)
        return cls, box, obj

    @staticmethod
    def __box_decode(anchors, box):
        xys = (box[..., :2] * anchors[..., 2:]) + anchors[..., :2]
        whs = box[..., 2:].exp() * anchors[..., 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        return torch.stack(tensors=[tl_x, tl_y, br_x, br_y], dim=-1)

    @staticmethod
    def __make_anchors(sizes, strides, device, dtype, offset=0.0):
        anchors = []
        assert len(sizes) == len(strides)
        for stride, size in zip(strides, sizes):
            # keep size as Tensor instead of int, so that we can convert to ONNX correctly
            shift_x = ((torch.arange(0, size[1]) + offset) * stride).to(dtype)
            shift_y = ((torch.arange(0, size[0]) + offset) * stride).to(dtype)

            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)
            stride_w = shift_x.new_full((shift_x.shape[0],), stride).to(dtype)
            stride_h = shift_x.new_full((shift_y.shape[0],), stride).to(dtype)
            anchors.append(torch.stack(tensors=[shift_x, shift_y, stride_w, stride_h], dim=-1).to(device))
        return anchors


class YOLOX(torch.nn.Module):
    def __init__(self, width, depth, num_classes):
        super().__init__()
        self.backbone = Backbone(width, depth)
        self.neck = Neck(width, depth)
        self.head = Head((width[3], width[4], width[5]), num_classes)

        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.head.stride = [256 / x.shape[-2] for x in self.forward(img_dummy)[0]]
        self.stride = self.head.stride

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, math.sqrt(5))

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def build(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLOX(width, depth, num_classes)
