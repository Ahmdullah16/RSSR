"""Small educational models; RGB and multispectral bands remain configurable."""
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(width, max(1,width//4),1), nn.ReLU(), nn.Conv2d(max(1,width//4),width,1), nn.Sigmoid())
        self.spatial = nn.Conv2d(2,1,7,padding=3)

    def forward(self, x):
        x = x * self.channel(x)
        return x * self.spatial(torch.cat([x.mean(1,keepdim=True),x.amax(1,keepdim=True)],1)).sigmoid()


class WindowBlock(nn.Module):
    def __init__(self, width, window=4):
        super().__init__()
        if width % 2 or window < 1:
            raise ValueError("Even width and positive window required")
        self.window = window
        self.position = nn.Parameter(torch.zeros(1,window*window,width))
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width,2,batch_first=True,dropout=0)
        self.ff = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))

    def forward(self, x):
        b,c,h,w = x.shape; s = self.window
        # Replicated padding is an explicit boundary convention, removed after attention.
        y = F.pad(x,(0,(-w)%s,0,(-h)%s),mode="replicate")
        H,W = y.shape[-2:]
        y = y.reshape(b,c,H//s,s,W//s,s).permute(0,2,4,3,5,1).reshape(-1,s*s,c)
        q = self.norm(y + self.position)
        y = y + self.attn(q,q,q,need_weights=False)[0]
        y = y + self.ff(y)
        return y.reshape(b,H//s,W//s,s,s,c).permute(0,5,1,3,2,4).reshape(b,c,H,W)[:,:,:h,:w]


class SRModel(nn.Module):
    def __init__(self, bands=3, scale=2, width=16, kind="hybrid", gate=False, fusion="concat", window=4, use_attention=True, constant_gate=1., zero_residual=False):
        super().__init__()
        if scale not in (2,4) or bands < 1 or width < 2 or kind not in ("srcnn","cnn","attention","transformer","hybrid") or fusion not in ("concat","add"):
            raise ValueError("Invalid model configuration")
        self.scale,self.bands,self.kind,self.fusion = scale,bands,kind,fusion
        if not 0<=constant_gate<=1 or (gate and constant_gate!=1.):
            raise ValueError('Constant gate must be in [0,1] and exclusive with learned gate')
        self.constant_gate=constant_gate
        self.srcnn = None
        if kind == "srcnn":
            self.context = None
            self.srcnn = nn.Sequential(nn.Conv2d(bands,32,9,padding=4),nn.ReLU(),nn.Conv2d(32,16,1),nn.ReLU(),nn.Conv2d(16,bands,5,padding=2))
            return
        self.head = nn.Conv2d(bands,width,3,padding=1)
        self.local = nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.ReLU(),nn.Conv2d(width,width,3,padding=1)) if kind != "transformer" else None
        self.context = WindowBlock(width,window) if kind in ("transformer","hybrid") else None
        self.attention = Attention(width) if use_attention and kind in ("attention","hybrid") else nn.Identity()
        self.fuse = nn.Conv2d(width*2,width,1) if kind == "hybrid" and fusion == "concat" else nn.Identity()
        self.tail = nn.Sequential(nn.Conv2d(width,bands*scale*scale,3,padding=1),nn.PixelShuffle(scale))
        if zero_residual:
            nn.init.zeros_(self.tail[0].weight); nn.init.zeros_(self.tail[0].bias)
        self.gate = nn.Sequential(nn.Conv2d(bands,width,3,padding=1),nn.ReLU(),nn.Conv2d(width,bands,1),nn.Sigmoid()) if gate else None

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.bands:
            raise ValueError("Expected NCHW with configured bands")
        base = F.interpolate(x,scale_factor=self.scale,mode="bicubic",align_corners=False)
        if self.srcnn is not None:
            return self.srcnn(base)
        f = self.head(x); local = f + self.local(f) if self.local is not None else f
        if self.kind == "transformer":
            f = self.context(f)
        elif self.kind == "hybrid":
            context = self.context(f)
            f = self.fuse(torch.cat([local,context],1)) if self.fusion == "concat" else (local+context)/2
        else:
            f = local
        residual = self.tail(self.attention(f))
        if self.gate is not None:
            residual = residual * F.interpolate(self.gate(x),scale_factor=self.scale,mode="nearest")
        return base + self.constant_gate*residual


@torch.inference_mode()
def tiled(model, x, tile=128, halo=32):
    """Bounded approximate tiling. Global channel pooling changes with tile context.

    Window grid origins align with the full image. Halo reduces convolution seams;
    do not claim exact equality for global channel-attention models.
    """
    if x.ndim != 4 or x.shape[0] != 1 or tile < 1 or halo < 0:
        raise ValueError("Tiling expects one NCHW image, positive tile, nonnegative halo")
    grid = model.context.window if model.context else 1
    tile = max(grid, tile//grid*grid); halo = ((halo+grid-1)//grid)*grid
    h,w = x.shape[-2:]; s = model.scale
    out = x.new_empty((1,model.bands,h*s,w*s))
    for y in range(0,h,tile):
        for z in range(0,w,tile):
            y0,z0 = max(0,y-halo),max(0,z-halo)
            y1,z1 = min(h,y+tile),min(w,z+tile)
            patch = model(x[:,:,y0:min(h,y1+halo),z0:min(w,z1+halo)])
            out[:,:,y*s:y1*s,z*s:z1*s] = patch[:,:,(y-y0)*s:(y1-y0)*s,(z-z0)*s:(z1-z0)*s]
    return out
