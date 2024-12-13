import math

from .utils_custom import *
from .yolo_bricks import RepDWConv, EffectiveSELayer


class ChannelC2f(BaseModule):

    def __init__(
            self,
            c1: int,
            c2: int,
            e: float = 1,
            n: int = 1,
            shortcut: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.c = int(c2 * e)
        self.cv1 = ConvModule(
            c1,
            2 * self.c,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.cv2 = ConvModule(
            (2 + n) * self.c,
            c2,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.m = nn.ModuleList(
            DarknetBottleneck(
                self.c,
                self.c,
                expansion=0.25,
                kernel_size=(3, 3),
                padding=(1, 1),
                add_identity=shortcut,
                use_depthwise=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(n))

    def forward(self, x: Tensor) -> Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class GatedFFN(nn.Module):  # 0608
    def __init__(self, c1: int,
                 c2: int,
                 n: int = 1,
                 shortcut: bool = False,
                 g: int = 1,
                 e: int = 3,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True)):
        super().__init__()
        self.n = n
        self.c = int(c2 * e)
        self.proj = ConvModule(c1, 2 * self.c,
                               kernel_size=1, stride=1, padding=0,
                               conv_cfg=conv_cfg,
                               norm_cfg=norm_cfg,
                               act_cfg=act_cfg)
        self.rep = RepDWConv(self.c, self.c)  # deploy=True
        self.m = nn.ModuleList(
            ConvModule(self.c, self.c, kernel_size=3, stride=1, padding=autopad(3), groups=self.c,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None) for _ in range(n - 1)
        )
        self.act = nn.GELU()
        self.cv2 = ConvModule(self.c, c2,
                              kernel_size=1, stride=1, padding=0,
                              conv_cfg=conv_cfg,
                              norm_cfg=norm_cfg,
                              act_cfg=None)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        shortcut = x.clone()
        x, z = self.proj(x).split([self.c, self.c], 1)
        x = self.rep(x)
        if self.n != 1:
            for m in self.m:
                x = m(x)
        x = x * self.act(z)
        x = self.cv2(x)
        return x + shortcut if self.add else x


class CED(BaseModule):
    def __init__(self, c1, c2, e=0.5,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True)):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvModule(c1, self.c, kernel_size=1,
                              stride=1, padding=0,
                              norm_cfg=norm_cfg,
                              act_cfg=act_cfg)
        self.cv2 = ConvModule(self.c * 4, c2, kernel_size=1,
                              stride=1, padding=0,
                              norm_cfg=norm_cfg,
                              act_cfg=None)

        # self.dwconv = nn.Sequential(RepDWConv(self.c, self.c), nn.SiLU())
        self.dwconv = ConvModule(self.c, self.c,
                                 kernel_size=3, stride=1, padding=autopad(3),
                                 groups=self.c,
                                 norm_cfg=norm_cfg,
                                 act_cfg=act_cfg)

    def forward(self, x):
        x = self.dwconv(self.cv1(x))
        x = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2],
        ], dim=1)
        x = self.cv2(x)
        return x






