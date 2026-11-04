"""Bounded minor-app operations. RGB HWC uint8, normalized CHW model tensors.

No downloads, model loading or training take place at import.
"""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SCALES = (2, 4, 8, 16, 32, 64)
MAX_SIDE = 1024
MAX_PIXELS = 1024 * 1024


def catalog():
    base=json.loads((ROOT/'configs/hosted-model.json').read_text())
    rest=json.loads((ROOT/'configs/hosted-models.json').read_text())['models']
    entries={'Bicubic':dict(label='Bicubic interpolation',limitations='Classical baseline; no learned checkpoint.'),
             'M1':base}
    for n in range(2,7):
        matches=[s for s in rest if s.get('architecture_id')==f'cat-v1/M{n}']
        if matches: entries[f'M{n}']=matches[0]
    hybrid=[s for s in rest if 'hybrid' in s['label'].lower()]
    if hybrid: entries['M7']=hybrid[0]
    return entries


def validate_image(image):
    a=np.asarray(image)
    if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 or not 1<=min(a.shape[:2]) or max(a.shape[:2])>MAX_SIDE:
        raise ValueError('Use uint8 RGB pixels, 1–1024 pixels per side.')
    return a


def preflight(image,scale):
    a=validate_image(image)
    if scale not in SCALES: raise ValueError('Supported scales: 2, 4, 8, 16, 32, 64.')
    h,w=a.shape[:2];ow,oh=w*scale,h*scale
    return dict(input_width=w,input_height=h,output_width=ow,output_height=oh,scale=scale,
                output_pixels=ow*oh,stages=int(np.log2(scale)),
                allowed=max(ow,oh)<=MAX_SIDE and ow*oh<=MAX_PIXELS,
                limit='Minor app: output <=1024 per side and <=1048576 pixels; no silent resizing.')


def image_hash(image):
    a=validate_image(image)
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()


def procedural_hr(size=128):
    """An explicitly synthetic resolution target, not satellite observations."""
    image=Image.new('RGB',(size,size),'#8ca86b');d=ImageDraw.Draw(image)
    d.rectangle((0,size//2,size,size//2+8),fill='#c9c7b5')
    d.rectangle((size//3,0,size//3+8,size),fill='#c9c7b5')
    for y in range(8,size-10,22):
        for x in range(8,size-10,25):
            d.rectangle((x,y,x+10,y+7),fill='#b35d3c',outline='#f0c9a6')
    for x in range(size//2,size):
        d.line((x,0,x,size//3),fill=(x*2%256,120,80))
    return np.asarray(image).copy()


def synthetic_pair(hr,scale,blur=0.5,noise=0.0,seed=17):
    hr=validate_image(hr);h,w=hr.shape[:2]
    if scale not in SCALES or h%scale or w%scale or min(h//scale,w//scale)<1:
        raise ValueError('Synthetic-reference width and height must be divisible by the selected scale. No crop is performed.')
    if not 0<=blur<=3 or not 0<=noise<=20: raise ValueError('Blur must be 0–3 HR pixels; noise 0–20 uint8 levels.')
    source=Image.fromarray(hr).filter(ImageFilter.GaussianBlur(blur))
    low=np.asarray(source.resize((w//scale,h//scale),Image.Resampling.BICUBIC)).astype(np.float32)
    low+=np.random.default_rng(seed).normal(0,noise,low.shape)
    return np.rint(low.clip(0,255)).astype(np.uint8),hr.copy()


def add_noise(image,sigma,seed=17):
    if not 0<=sigma<=20: raise ValueError('Noise must be 0–20 uint8 levels.')
    a=validate_image(image).astype(np.float32)
    return np.rint(np.clip(a+np.random.default_rng(seed).normal(0,sigma,a.shape),0,255)).astype(np.uint8)


def encode_image(image,format='PNG',quality=95):
    a=validate_image(image)
    if format not in ('PNG','JPEG'): raise ValueError('Choose PNG or JPEG.')
    if not 75<=quality<=100: raise ValueError('JPEG quality must be 75–100.')
    out=io.BytesIO();kwargs={'quality':quality,'subsampling':0} if format=='JPEG' else {}
    Image.fromarray(a).save(out,format=format,**kwargs)
    return out.getvalue()


def finite_json(value):
    if isinstance(value,dict): return {str(k):finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [finite_json(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value): return 'infinity' if np.isinf(value) else None
    if isinstance(value,np.generic): return value.item()
    return value


def evaluate(output,reference=None,border=0):
    output=validate_image(output)
    gray=output.astype(np.float32).mean(axis=2)/255
    differences=[]
    if gray.shape[0]>1: differences.append(np.mean(np.abs(np.diff(gray,axis=0))))
    if gray.shape[1]>1: differences.append(np.mean(np.abs(np.diff(gray,axis=1))))
    answer={'edge_activity':float(np.mean(differences)) if differences else 0.,
            'reference_available':reference is not None}
    if reference is None: return answer
    reference=validate_image(reference)
    if reference.shape!=output.shape: raise ValueError('Aligned HR reference must exactly match the output shape; it is never resized automatically.')
    if not isinstance(border,int) or border<0 or 2*border>=min(output.shape[:2]): raise ValueError('Metric border crop is too large for this image.')
    from rssr.core import metrics
    import torch
    tensor=lambda a:torch.from_numpy(a.astype(np.float32)/255).permute(2,0,1)
    scores=metrics(tensor(output),tensor(reference),data_range=1.,border=border)
    answer.update({key:scores[key] for key in ('mse','rmse','psnr','ssim')})
    return answer


def infer(image,model,scale,tile=128,halo=32):
    """Repeat native x2 inference; clip/quantize each stage, matching display scope."""
    request=preflight(image,scale)
    if not request['allowed']: raise ValueError(request['limit'])
    if tile not in (64,128,256) or halo not in (16,32,64): raise ValueError('Unsupported tile/halo setting.')
    import torch
    from rssr.models import tiled
    if model.bands!=3 or model.scale!=2: raise ValueError('Require a trained native x2 RGB checkpoint.')
    torch.set_num_threads(2);model.eval();pixels=image.copy();stages=[]
    started=time.perf_counter()
    with torch.inference_mode():
        for stage in range(request['stages']):
            x=torch.from_numpy(pixels.astype(np.float32)/255).permute(2,0,1)[None]
            y=tiled(model,x,tile=tile,halo=halo)[0]
            if y.shape!=(3,pixels.shape[0]*2,pixels.shape[1]*2) or not torch.isfinite(y).all():
                raise ValueError('Model returned invalid values or dimensions.')
            stages.append({'scale':2**(stage+1),'clipped_fraction':float(((y<0)|(y>1)).float().mean())})
            pixels=np.rint(y.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    return pixels,{'runtime_seconds':time.perf_counter()-started,'tile':tile,'halo':halo,'stages':stages,
                   'tiling_note':'Core-trimmed halo tiling; global attention depends on tile context.',
                   'scale_note':'Native learned x2; higher scales are repeated x2 stages, not separately trained checkpoints.'}


def bicubic(image,scale):
    request=preflight(image,scale)
    if not request['allowed']: raise ValueError(request['limit'])
    return np.asarray(Image.fromarray(image).resize((request['output_width'],request['output_height']),Image.Resampling.BICUBIC)).copy()


def result_bundle(results,format='PNG',quality=95):
    out=io.BytesIO();extension='png' if format=='PNG' else 'jpg'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
        for i,result in enumerate(results,1):
            archive.writestr(f'{i}-{result["model_id"]}-x{result["scale"]}.{extension}',encode_image(result['pixels'],format,quality))
            archive.writestr(f'{i}-provenance.json',json.dumps(finite_json({**result['metadata'],'export':{'format':format,'jpeg_quality':quality if format=='JPEG' else None,'metrics_before_compression':True}}),indent=2,allow_nan=False))
    return out.getvalue()
