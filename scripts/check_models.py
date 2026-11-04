"""Bounded real-checkpoint check. Downloads <=7*262144 bytes; no training."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import json,time,tempfile
import torch
import numpy as np
from minor_app import service as s
from rssr.hosted_assets import fetch_asset
from rssr.model_factory import from_config


def main():
    torch.set_num_threads(2)
    image=np.random.default_rng(17).integers(0,256,(2,3,3),dtype=np.uint8)
    records=[]
    for key,spec in s.catalog().items():
        if key=='Bicubic': continue
        path=fetch_asset(spec['checkpoint'],Path(tempfile.gettempdir())/'rssr-minor-checkpoints-v1')
        state=torch.load(path,map_location='cpu',weights_only=True)
        model=from_config(state['config']['model']);model.load_state_dict(state['model'])
        for scale in s.SCALES:
            output,meta=s.infer(image,model,scale)
            assert output.shape==(2*scale,3*scale,3)
            assert output.dtype==np.uint8
            records.append(dict(model_id=key,checkpoint_sha256=spec['checkpoint']['sha256'],scale=scale,output_shape=list(output.shape),runtime_seconds=meta['runtime_seconds'],status='passed'))
    report={'scope':'Tiny 2x3 RGB shape/finite-value engineering checks with real distinct pinned checkpoints; not image-quality validation.',
            'cases':records,'torch_version':torch.__version__,'passed':len(records)}
    target=ROOT/'reports/model_validation.json'
    target.write_text(json.dumps(report,indent=2))
    print(f'{len(records)} real-checkpoint scale cases passed')

if __name__=='__main__': main()
