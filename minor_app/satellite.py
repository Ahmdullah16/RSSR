"""Explicit, bounded Sentinel-2 RGB-display acquisition; no inference or training."""
from datetime import datetime,timezone
import json
import urllib.request
import numpy as np

ENDPOINT='https://earth-search.aws.element84.com/v1/search'
COLLECTIONS=('sentinel-2-c1-l2a','sentinel-2-l2a')


def query_request(bbox,start,end,cloud,collection,crop_size):
    if len(bbox)!=4 or not np.isfinite(bbox).all(): raise ValueError('Enter four finite WGS84 coordinates.')
    west,south,east,north=map(float,bbox)
    if not -180<=west<east<=180 or not -90<=south<north<=90: raise ValueError('Require west < east and south < north in valid longitude/latitude ranges.')
    if east-west>.25 or north-south>.25: raise ValueError('Minor-app AOI edges must be at most 0.25 degrees.')
    if start>end or (end-start).days>366: raise ValueError('Use an ordered date interval no longer than 366 days.')
    if not 0<=cloud<=100 or collection not in COLLECTIONS or crop_size not in (64,128,256): raise ValueError('Unsupported cloud limit, collection or crop size.')
    return dict(bbox=[west,south,east,north],datetime=f'{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z',
                collections=[collection],limit=1,sortby=[{'field':'properties.datetime','direction':'desc'}],
                query={'eo:cloud_cover':{'lte':cloud}})


def fetch_json(body):
    request=urllib.request.Request(ENDPOINT,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=30) as response:
        payload=response.read(2_000_001)
        if len(payload)>2_000_000: raise ValueError('STAC response exceeded 2 MB metadata budget.')
        return json.loads(payload)


def acquire(bbox,start,end,cloud,collection,crop_size,fetch=fetch_json,cog_context=None):
    body=query_request(bbox,start,end,cloud,collection,crop_size)
    result=fetch(body)
    if not result.get('features'): raise ValueError('No Sentinel observation matches these dates, AOI and cloud limit. Widen dates or change the cloud limit.')
    item=result['features'][0];asset=item.get('assets',{}).get('visual')
    if not asset: raise ValueError('This item has no compatible true-color visual asset.')
    if cog_context is None:
        from rssr.cog import bounded_cog
        cog_context=bounded_cog
    import rasterio
    from rasterio.windows import Window
    from rasterio.warp import transform_bounds
    with cog_context(asset['href'],32_000_000) as (url,transfer):
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_MULTIRANGE='SERIAL',GDAL_HTTP_MAX_RETRY='0'):
            with rasterio.open(url) as ds:
                if ds.count!=3 or ds.dtypes!=('uint8',)*3 or not ds.crs: raise ValueError('The provider asset is not georeferenced uint8 RGB display imagery.')
                left,bottom,right,top=transform_bounds('EPSG:4326',ds.crs,*body['bbox'])
                bounds=ds.bounds
                left,bottom,right,top=max(left,bounds.left),max(bottom,bounds.bottom),min(right,bounds.right),min(top,bounds.top)
                if left>=right or bottom>=top: raise ValueError('Requested AOI does not overlap the visual asset.')
                if min(ds.width,ds.height)<crop_size: raise ValueError('Asset is smaller than the requested crop.')
                row,col=ds.index((left+right)/2,(bottom+top)/2)
                window=Window(max(0,min(ds.width-crop_size,col-crop_size//2)),max(0,min(ds.height-crop_size,row-crop_size//2)),crop_size,crop_size)
                a=ds.read(window=window);mask=(ds.read_masks(window=window)>0).all(axis=0)
                if a.shape!=(3,crop_size,crop_size) or mask.mean()<.8: raise ValueError('Crop is incomplete or has less than 80% valid pixels.')
                a[:,~mask]=0
                transform=ds.window_transform(window)
                crs=str(ds.crs);pixel_size=list(ds.res)
    props=item.get('properties',{})
    metadata={'kind':'live Sentinel-2 query','provider':'Element 84 Earth Search / public Sentinel COG archive',
              'collection':collection,'satellite_platform':props.get('platform','Not supplied by provider'),
              'constellation':props.get('constellation','Sentinel-2'),'instrument':props.get('instruments',['MSI (Sentinel-2 mission)']),
              'provider_item_id':item['id'],'acquisition_time':props.get('datetime','Unknown'),
              'retrieval_utc':datetime.now(timezone.utc).isoformat(),'asset_url':asset['href'],
              'bands':'RGB true-color display based on Sentinel-2 B04/B03/B02; provider-stretched uint8, not reflectance',
              'cloud_cover_percent':props.get('eo:cloud_cover'),'cloud_filter':'Scene-level only; not a pixel cloud mask',
              'query':body,'crop_note':'Fixed-size native-pixel crop centered on the AOI/scene intersection; not the full AOI extent.',
              'crop_crs':crs,'crop_transform':list(transform),'source_pixel_size_crs_units':pixel_size,
              'valid_fraction':float(mask.mean()),'invalid_pixel_handling':'Invalid source pixels are displayed black; PNG/JPEG exports do not retain the mask.',
              'license_links':[link['href'] for link in item.get('links',[]) if link.get('rel')=='license'],
              'attribution':'Contains modified Copernicus Sentinel data; Element 84 COG archive',
              'transfer':transfer,'reference_available':False,'calibration':'Display RGB only; no reflectance-calibrated or multispectral processing.',
              'processing_level':'L2A source (atmospherically corrected); selected visual product is display encoded.',
              'quality_limitation':'NAIP-trained SR pilots applied cross-sensor; Sentinel reconstruction fidelity unvalidated.'}
    return a.transpose(1,2,0).copy(),metadata
