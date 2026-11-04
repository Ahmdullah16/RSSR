"""Versioned C/A/T factory. Legacy SRModel names/checkpoint layouts stay unchanged."""
from dataclasses import dataclass
from torch import nn
from .models import SRModel, Attention


@dataclass(frozen=True)
class Architecture:
    id: str
    c: bool
    a: bool
    t: bool
    legacy: str | None


REGISTRY = {
    item.id: item for item in (
        Architecture('cat-v1/M1', True, False, False, 'cnn'),
        Architecture('cat-v1/M2', False, True, False, None),
        Architecture('cat-v1/M3', False, False, True, 'transformer'),
        Architecture('cat-v1/M4', True, True, False, 'attention'),
        Architecture('cat-v1/M5', True, False, True, 'hybrid:use_attention=False'),
        Architecture('cat-v1/M6', False, True, True, None),
        Architecture('cat-v1/M7', True, True, True, 'hybrid:use_attention=True'),
    )
}


def build(architecture_id, *, bands=3, scale=2, width=16, window=4,
          zero_residual=False):
    """Return an untrained trainable model, never a downloaded checkpoint.

    C means the two-convolution residual local branch; A means additional
    channel/spatial gating; T means window self-attention and feed-forward block.
    All variants retain convolutional embedding/reconstruction and bicubic base.
    """
    if architecture_id not in REGISTRY:
        raise ValueError('Unknown versioned architecture ID')
    spec = REGISTRY[architecture_id]
    kind = 'hybrid' if spec.c and spec.t else 'transformer' if spec.t else 'cnn'
    model = SRModel(bands=bands, scale=scale, width=width, kind=kind,
                    window=window, use_attention=False, zero_residual=zero_residual)
    if not spec.c:
        model.local = None
    model.attention = Attention(width) if spec.a else nn.Identity()
    model.architecture_id = spec.id
    return model


def describe(architecture_id, **config):
    spec = REGISTRY[architecture_id]
    model = build(architecture_id, **config)
    return {'architecture_id': spec.id, 'modules': {'C': spec.c, 'A': spec.a, 'T': spec.t},
            'config': {'bands': 3, 'scale': 2, 'width': 16, 'window': 4,
                       'zero_residual': False, **config},
            'parameters': sum(p.numel() for p in model.parameters()),
            'legacy_structure': spec.legacy,
            'checkpoint_sha256': None, 'training_provenance': None,
            'status': {'implemented': True, 'trained': False, 'evaluated': False,
                       'deployed': False},
            'status_scope': 'This registry configuration; historical runs require explicit provenance binding.',
            'native_scales_implemented': [2, 4],
            'bands_implemented': 'positive integer; trained compatibility unverified',
            'tiling': 'A uses tile-dependent global channel pooling; T requires aligned windows.'}
