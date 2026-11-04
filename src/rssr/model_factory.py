"""Explicit routing for legacy and versioned model configurations."""
from .models import SRModel
from .architecture_registry import build


def from_config(config):
    config = dict(config)
    if 'architecture_id' in config:
        if 'kind' in config:
            raise ValueError('Choose architecture_id or legacy kind, not both')
        return build(config.pop('architecture_id'), **config)
    return SRModel(**config)
