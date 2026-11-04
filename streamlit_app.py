"""Independent RSSR minor app. Community Cloud main file: streamlit_app.py."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import hashlib
import io
import json
import tempfile
import threading
import time
from datetime import datetime,timezone,timedelta
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from minor_app import service as svc
from minor_app.theory import render as render_theory
from rssr.uploads import decode_image

st.set_page_config(page_title='RSSR | Minor Project Lab',page_icon='🛰️',layout='wide')
st.markdown('''<style>.block-container{max-width:1250px;padding-top:2rem}
[data-testid="stMetric"]{border:1px solid #cbdde8;border-radius:12px;padding:.8rem;background:rgba(20,125,146,.06)}
h1,h2,h3{letter-spacing:-.02em}</style>''',unsafe_allow_html=True)
CATALOG=svc.catalog()
FAMILIES={'Bicubic':'Bicubic baseline','M1':'CNN','M2':'Attention','M3':'Transformer','M4':'CNN + Attention','M5':'CNN + Transformer','M6':'Attention + Transformer','M7':'CNN + Attention + Transformer'}
CACHE=Path(tempfile.gettempdir())/'rssr-minor-checkpoints-v1'

@st.cache_resource
def cpu_lock(): return threading.Lock()

@st.cache_resource(max_entries=2,show_spinner=False)
def load_resource(model_id):
    import torch
    from rssr.hosted_assets import fetch_asset
    from rssr.model_factory import from_config
    spec=CATALOG[model_id]
    path=fetch_asset(spec['checkpoint'],CACHE)
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=from_config(state['config']['model']);model.load_state_dict(state['model']);model.eval()
    if model.scale!=2 or model.bands!=3: raise ValueError('This app requires a native ×2 RGB checkpoint.')
    return model,{'run_id':state.get('run_id'),'training_provenance':state.get('provenance'),
                  'model_config':state['config']['model'],'parameter_count':sum(p.numel() for p in model.parameters())}

@st.cache_data(max_entries=6,show_spinner=False)
def compute(image,reference,model_id,scale,tile,halo,border,source):
    plan=svc.preflight(image,scale)
    if not plan['allowed']: raise ValueError(plan['limit'])
    if reference is not None and reference.shape!=(plan['output_height'],plan['output_width'],3):
        raise ValueError('Aligned HR dimensions must exactly equal the output dimensions. No automatic alignment/resizing is applied.')
    started=time.perf_counter()
    if model_id=='Bicubic':
        output=svc.bicubic(image,scale);extra={'method':'Pillow bicubic interpolation, not learned inference.'}
    else:
        model,info=load_resource(model_id)
        lock=cpu_lock()
        if not lock.acquire(blocking=False): raise ValueError('CPU is busy with another request. Retry shortly.')
        try: output,extra=svc.infer(image,model,scale,tile,halo)
        finally: lock.release()
        extra.update(info)
    score=svc.evaluate(output,reference,border)
    metadata={'application':'RSSR minor lab v1','model_id':model_id,'model':CATALOG[model_id],
              'input_pixel_sha256':svc.image_hash(image),'reference_pixel_sha256':svc.image_hash(reference) if reference is not None else None,
              'source':source,'request':plan,'metrics':score,'metric_border':border,'metric_range':1.,
              'created_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-started,
              'output_encoding':'RGB uint8 display; metrics before JPEG compression','storage':'Temporary session/cache only',**extra}
    return {'pixels':output,'model_id':model_id,'scale':scale,'metadata':metadata}


def signature(image,reference,settings):
    payload=[svc.image_hash(image),svc.image_hash(reference) if reference is not None else None,settings]
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def score_row(result):
    m=result['metadata'];s=m['metrics']
    return {'Model':FAMILIES[result['model_id']],'Scale':result['scale'],'Runtime (s)':m['elapsed_seconds'],
            'PSNR (dB)':s.get('psnr'),'SSIM':s.get('ssim'),'RMSE':s.get('rmse'),'Edge activity':s['edge_activity']}


def display_result(result,reference,baseline):
    a=result['pixels']; columns=st.columns(2)
    columns[0].image(a,caption=f'{FAMILIES[result["model_id"]]} · {a.shape[1]}×{a.shape[0]}',width='stretch')
    comparator=reference if reference is not None else baseline
    delta=np.abs(a.astype(np.float32)-comparator.astype(np.float32)).mean(axis=2)
    columns[1].image(np.rint(delta).clip(0,255).astype(np.uint8),caption='Absolute HR error (0–255 display levels)' if reference is not None else 'Absolute disagreement from bicubic (not HR error)',clamp=True,width='stretch')
    st.dataframe(pd.DataFrame([score_row(result)]),hide_index=True,width='stretch')
    st.caption('PSNR: reference similarity in dB, higher is better. SSIM: reference structural similarity, closer to 1 is better. RMSE: normalized RGB error, lower is better. Edge activity: texture/contrast, not verified fidelity. Runtime: original computation time; a cache hit reuses that recorded timing.')
    if reference is None: st.caption('PSNR/SSIM/RMSE are unavailable without an aligned HR reference. Edge activity alone does not establish quality.')

with st.sidebar:
    st.header('Input & controls')
    source_mode=st.selectbox('Input source',['Synthetic demonstration','Live satellite (Sentinel-2)','Upload LR image','Upload HR for synthetic experiment','Licensed NAIP training example'],key='source_mode')
    model_id=st.selectbox('Model',list(CATALOG),format_func=lambda key:f'{key} · {FAMILIES[key]}',key='model_id')
    scale=st.selectbox('Upscaling factor',svc.SCALES,format_func=lambda s:f'×{s}',key='scale')
    blur=st.slider('Synthetic blur σ (HR pixels)',0.,3.,.5,.1,key='blur')
    noise=st.slider('Input noise σ (uint8 levels)',0.,20.,0.,1.,key='noise')
    seed=st.number_input('Noise seed',0,2147483647,17,1,key='seed')
    tile=st.selectbox('Tile size (LR pixels)',[64,128,256],index=1,key='tile')
    halo=st.selectbox('Halo context (LR pixels)',[16,32,64],index=1,key='halo')
    border=st.number_input('Metric border crop (HR pixels)',0,16,0,1,key='border')
    export_format=st.selectbox('Download image format',['PNG','JPEG'],key='export_format')
    quality=st.slider('JPEG quality',75,100,95,key='jpeg_quality',disabled=export_format!='JPEG')
    st.caption('Tile and halo affect learned models; Bicubic ignores them. Blur applies only to synthetic-pair sources.')
    st.caption('Current results recalculate on committed control changes. Compare and sensitivity batches run only when requested to bound CPU use.')
    st.caption('Input ≤8 MB, ≤1024 per side. Output ≤1024 per side and ≤1,048,576 pixels. Large requests are rejected, never silently resized.')

st.caption('REMOTE SENSING SUPER-RESOLUTION · SEPARATE MINOR PROJECT')
st.title('RSSR · Image Reconstruction Lab')
st.markdown('CNN, attention and Transformer pilots in a compact five-section workflow. Upload an image, explore parameter effects, compare models and export PNG or JPEG.')
st.caption('Prepared for Ahmdullah · M.Tech, Signal Processing and Machine Learning · IIT Guwahati')
page=st.radio('Sections',['Results','Sensitivity','Compare','Concepts & mathematics','Downloads'],horizontal=True,label_visibility='collapsed',key='page')
if page=='Concepts & mathematics':
    render_theory()

image=reference=hr=clean_lr=None
source={'kind':source_mode}
try:
    if source_mode=='Synthetic demonstration':
        hr=svc.procedural_hr(128)
        image,reference=svc.synthetic_pair(hr,scale,blur,noise,int(seed))
        source.update(note='Procedural synthetic test pattern, not satellite observations.',blur_hr_pixels=blur,noise_uint8=noise,seed=int(seed))
    elif source_mode in ('Upload LR image','Upload HR for synthetic experiment'):
        upload=st.file_uploader('Upload PNG / JPG / JPEG',type=['png','jpg','jpeg'],key='upload_image')
        if upload is None: st.info('Choose an image to begin.');st.stop()
        source.update(filename=upload.name,georeferencing='Not available in this RGB workflow')
        original=decode_image(upload.getvalue())
        if source_mode=='Upload HR for synthetic experiment':
            hr=original;image,reference=svc.synthetic_pair(hr,scale,blur,noise,int(seed))
            source.update(blur_hr_pixels=blur,noise_uint8=noise,seed=int(seed))
        else:
            clean_lr=original;image=svc.add_noise(clean_lr,noise,int(seed))
            source.update(noise_uint8=noise,seed=int(seed),blur='not applied to LR uploads')
            with st.expander('Optional upload sensor information'):
                source['sensor_user_supplied']=st.text_input('Sensor / satellite (if known)','Unknown',key='upload_sensor')
                source['acquisition_user_supplied']=st.text_input('Capture date/time (if known)','Unknown',key='upload_capture')
                st.caption('These values are user supplied and are not verified from RGB file content.')
            aligned=st.file_uploader('Optional aligned HR reference, exact output dimensions',type=['png','jpg','jpeg'],key='hr_reference')
            if aligned: reference=decode_image(aligned.getvalue());source['alignment']='User asserted; not independently verified'
    elif source_mode=='Live satellite (Sentinel-2)':
        from minor_app.satellite import acquire,COLLECTIONS
        st.subheader('Live Sentinel-2 image request')
        st.markdown('**Satellite:** Copernicus Sentinel-2 constellation, **instrument:** MSI. The exact spacecraft, item ID and capture time are reported from the selected observation. RGB uses the provider’s B04/B03/B02 true-color display product. A live request can return an older acquisition; it is not a live camera.')
        bbox_text=st.text_input('AOI west, south, east, north (WGS84 degrees)','78.035,27.165,78.055,27.185',key='live_bbox',help='Small Taj Mahal area by default. West/east are longitudes; south/north are latitudes.')
        cols=st.columns(2)
        today=datetime.now(timezone.utc).date()
        start=cols[0].date_input('Acquisition start',today-timedelta(days=90),key='live_start')
        end=cols[1].date_input('Acquisition end',today,key='live_end')
        cloud=st.slider('Maximum scene cloud cover (%)',0,100,30,key='live_cloud')
        collection=st.selectbox('Sentinel collection',COLLECTIONS,key='live_collection')
        crop_size=st.selectbox('Native crop side (pixels)',[64,128,256],index=1,key='live_crop')
        st.caption('Cloud limit filters the whole scene, not clouds over this exact crop. Dates constrain capture time. Crop size is a square at native source sampling, centered on the requested AOI. Download budget: 32 MB per request.')
        live_key=json.dumps([bbox_text,str(start),str(end),cloud,collection,crop_size])
        if st.button('Acquire satellite image',key='live_acquire'):
            bounds=[float(value.strip()) for value in bbox_text.split(',')]
            with st.spinner('Searching and downloading a bounded native RGB crop…'):
                raw,metadata=acquire(bounds,start,end,cloud,collection,crop_size)
            st.session_state['live_input']={'key':live_key,'image':raw,'source':metadata}
        live=st.session_state.get('live_input')
        if not live or live['key']!=live_key:
            st.info('Acquire the satellite crop for these query settings. Older results are hidden after query changes.');st.stop()
        clean_lr=live['image'];image=svc.add_noise(clean_lr,noise,int(seed));source=live['source'].copy()
        source.update(added_noise_uint8=noise,noise_seed=int(seed))
        st.caption(f'Spacecraft: {source["satellite_platform"]} · capture: {source["acquisition_time"]} · item: {source["provider_item_id"]}')
    else:
        spec=CATALOG['M1']['sample']
        st.caption(f'Cached NAIP training example · acquisition {spec["acquisition_time"]} · {spec["license"]}')
        if st.button('Download and load licensed example',key='load_naip'):
            from rssr.hosted_assets import fetch_asset
            st.session_state['naip_bytes']=fetch_asset(spec,CACHE).read_bytes()
        if 'naip_bytes' not in st.session_state: st.info('Load the example or choose an upload/synthetic input.');st.stop()
        clean_lr=decode_image(st.session_state['naip_bytes']);image=svc.add_noise(clean_lr,noise,int(seed));source.update(spec)
    request=svc.preflight(image,scale)
    st.caption(f'Input {request["input_width"]}×{request["input_height"]} → output {request["output_width"]}×{request["output_height"]} · {request["output_pixels"]:,} pixels · {request["stages"]} ×2 stage(s)')
    if not request['allowed']: st.error(request['limit']);st.stop()
    if 2*border>=min(request['output_width'],request['output_height']): st.error('Metric border removes the entire image; reduce border crop.');st.stop()
    if scale>2: st.caption('Higher scales repeat a native ×2 model. Additional pixels and apparent sharpness are not verified recovered geographic detail.')
    st.caption(CATALOG[model_id]['limitations'])
    with st.expander('Input provenance, parameter meanings & current values',expanded=source_mode in ('Live satellite (Sentinel-2)','Upload LR image')):
        st.dataframe(pd.DataFrame([
            {'Parameter':'Model','Current value':FAMILIES[model_id],'Meaning':'Selects fixed trained weights; Bicubic is interpolation only.'},
            {'Parameter':'Scale','Current value':f'×{scale}','Meaning':'Multiplies width and height; total pixels grow by scale squared.'},
            {'Parameter':'Blur σ','Current value':str(blur),'Meaning':'HR-pixel blur used only for synthetic-pair creation; not applied to real LR inputs.'},
            {'Parameter':'Noise σ','Current value':str(noise),'Meaning':'Standard deviation of added Gaussian noise, in 0–255 display levels.'},
            {'Parameter':'Tile / halo','Current value':f'{tile} / {halo}','Meaning':'LR processing tile and surrounding context; affects memory and attention context.'},
            {'Parameter':'Metric crop','Current value':str(border),'Meaning':'HR border pixels excluded from reference scores only; does not crop output images.'},
            {'Parameter':'JPEG quality','Current value':str(quality),'Meaning':'Export compression quality; metrics are measured before compression.'}
        ]),hide_index=True,width='stretch')
        st.json(svc.finite_json(source))

    if page=='Concepts & mathematics': st.stop()
    if page=='Results':
        with st.spinner('Reconstructing and measuring the current input…'):
            result=compute(image,reference,model_id,scale,tile,halo,int(border),source)
        baseline=svc.bicubic(image,scale)
        cols=st.columns(2);cols[0].image(image,caption='Current LR input',width='stretch');cols[1].image(reference if reference is not None else baseline,caption='Known synthetic/aligned HR reference' if reference is not None else 'Bicubic comparison baseline',width='stretch')
        display_result(result,reference,baseline)
        fmt=export_format;ext='png' if fmt=='PNG' else 'jpg'
        st.download_button(f'Download current SR image ({fmt})',svc.encode_image(result['pixels'],fmt,quality),f'RSSR-{model_id}-x{scale}.{ext}','image/png' if fmt=='PNG' else 'image/jpeg')
        with st.expander('Checkpoint, architecture and complete run metadata'): st.json(svc.finite_json(result['metadata']))

    elif page=='Compare':
        selected=st.multiselect('Models to compare (maximum 3)',list(CATALOG),default=['Bicubic','M1','M7'],format_func=lambda key:f'{key} · {FAMILIES[key]}',max_selections=3,key='compare_models')
        key=signature(image,reference,dict(models=selected,scale=scale,tile=tile,halo=halo,border=border,source=source))
        if st.button('Run model comparison',disabled=not selected,key='run_compare'):
            results=[]
            for choice in selected:
                with st.spinner(f'Running {choice}…'): results.append(compute(image,reference,choice,scale,tile,halo,int(border),source))
            st.session_state['comparison']={'key':key,'results':results}
        batch=st.session_state.get('comparison')
        if batch and batch['key']==key:
            st.dataframe(pd.DataFrame([score_row(r) for r in batch['results']]),hide_index=True,width='stretch')
            for col,r in zip(st.columns(len(batch['results'])),batch['results']): col.image(r['pixels'],caption=f'{r["model_id"]} · ×{r["scale"]}',width='stretch')
            st.download_button('Download comparison images & metadata',svc.result_bundle(batch['results'],export_format,quality),'RSSR-comparison.zip','application/zip')
            st.warning('Checkpoint training budgets differ. This comparison demonstrates outputs; it does not establish a controlled architecture ranking.')
        else: st.info('Run the selected models. Changing any input clears the displayed comparison until recomputed.')

    elif page=='Sensitivity':
        axis=st.selectbox('Vary one setting',['Scale','Tile size','Input noise'],key='sensitivity_axis')
        values=[2,4,8] if axis=='Scale' else [64,128,256] if axis=='Tile size' else [0.,5.,10.]
        st.caption(f'Three-case experiment: {axis} = {values}. Other controls and the checkpoint stay fixed. Scale experiments rebuild synthetic LR from the same HR when that source mode is active.')
        key=signature(image,reference,dict(axis=axis,model=model_id,scale=scale,tile=tile,halo=halo,border=border,blur=blur,seed=int(seed),source=source))
        if st.button('Run sensitivity experiment',key='run_sensitivity'):
            rows=[];results=[]
            for value in values:
                try:
                    s=value if axis=='Scale' else scale;t=value if axis=='Tile size' else tile
                    if hr is not None: low,ref=svc.synthetic_pair(hr,s,blur,value if axis=='Input noise' else noise,int(seed))
                    else: low,ref=(svc.add_noise(clean_lr,value,int(seed)),reference) if axis=='Input noise' else (image,reference)
                    provenance=dict(source,sensitivity_setting=axis,sensitivity_value=value)
                    if axis=='Input noise':
                        provenance['noise_uint8']=value
                        if 'added_noise_uint8' in provenance: provenance['added_noise_uint8']=value
                    r=compute(low,ref,model_id,s,t,halo,int(border),provenance)
                    rows.append(dict(Setting=value,Status='complete',**score_row(r)));results.append(r)
                except (ValueError,RuntimeError) as error: rows.append(dict(Setting=value,Status=str(error)))
            st.session_state['sensitivity']={'key':key,'rows':rows,'results':results}
        batch=st.session_state.get('sensitivity')
        if batch and batch['key']==key:
            frame=pd.DataFrame(batch['rows']);st.dataframe(frame,hide_index=True,width='stretch')
            columns=[c for c in ['PSNR (dB)','SSIM','Edge activity'] if c in frame and pd.to_numeric(frame[c],errors='coerce').notna().any()]
            for column in columns: st.line_chart(frame.set_index('Setting')[[column]])
            if batch['results']:
                for col,r in zip(st.columns(len(batch['results'])),batch['results']): col.image(r['pixels'],caption=str(r['metadata']['source']['sensitivity_value']),width='stretch')
                st.download_button('Download sensitivity images & provenance',svc.result_bundle(batch['results'],export_format,quality),'RSSR-sensitivity.zip','application/zip')
            st.download_button('Download sensitivity table',frame.to_csv(index=False),'sensitivity.csv','text/csv')
        else: st.info('Run the three bounded cases to quantify the selected parameter’s effect.')

    else:
        result=compute(image,reference,model_id,scale,tile,halo,int(border),source)
        st.subheader('Current output downloads')
        for fmt,ext,mime in [('PNG','png','image/png'),('JPEG','jpg','image/jpeg')]:
            st.download_button(f'Super-resolved image ({fmt})',svc.encode_image(result['pixels'],fmt,quality),f'RSSR-{model_id}-x{scale}.{ext}',mime,key='download_'+fmt)
        st.download_button('Run provenance (JSON)',json.dumps(svc.finite_json(result['metadata']),indent=2,allow_nan=False),'RSSR-provenance.json','application/json')
        st.download_button('Measured results (CSV)',pd.DataFrame([score_row(result)]).to_csv(index=False),'RSSR-results.csv','text/csv')
        st.download_button('Image + provenance bundle (ZIP)',svc.result_bundle([result],export_format,quality),'RSSR-output.zip','application/zip')
        st.caption('Metrics describe the reconstructed pixels before JPEG compression. PNG is lossless; JPEG is lossy. Both are RGB display products without geographic CRS.')
except Exception as error:
    st.error(f'Request did not complete: {error}')
    st.caption('No trained output is substituted when a checkpoint or provider is unavailable. Choose Bicubic for the offline baseline, or retry the selected model later.')
st.divider();st.caption('RSSR Minor Lab · independent deployment · temporary image processing · seven genuine pilot families')
