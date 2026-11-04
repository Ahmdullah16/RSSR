from contextlib import contextmanager
from datetime import date
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from minor_app.satellite import acquire,query_request


def test_query_bounds_and_dates():
    args=([78.,27.,78.1,27.1],date(2026,1,1),date(2026,1,30),20,'sentinel-2-l2a',64)
    query=query_request(*args)
    assert query['limit']==1
    assert query['collections']==['sentinel-2-l2a']
    with pytest.raises(ValueError): query_request([78,27,79,28],*args[1:])
    with pytest.raises(ValueError): query_request(args[0],args[2],args[1],*args[3:])
    with pytest.raises(ValueError): query_request(args[0],args[1],args[2],101,*args[4:])


def test_bounded_acquisition_metadata_and_pixels(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    path=tmp_path/'rgb.tif'
    pixels=np.full((3,128,128),100,np.uint8)
    with rasterio.open(path,'w',driver='GTiff',width=128,height=128,count=3,dtype='uint8',crs='EPSG:4326',transform=from_origin(78.,27.2,.001,.001)) as ds: ds.write(pixels)
    @contextmanager
    def local_asset(url,budget):
        assert budget==32_000_000
        yield str(path),{'budget_bytes':budget,'bytes_received':100}
    def fetch(body):
        return {'features':[{'id':'test-item','properties':{'platform':'sentinel-2c','datetime':'2026-01-15T00:00:00Z','eo:cloud_cover':12},'assets':{'visual':{'href':'https://example.invalid/visual.tif'}},'links':[{'rel':'license','href':'https://example.invalid/license'}]}]}
    out,meta=acquire([78.02,27.1,78.06,27.15],date(2026,1,1),date(2026,1,30),20,'sentinel-2-l2a',64,fetch=fetch,cog_context=local_asset)
    assert out.shape==(64,64,3)
    assert meta['satellite_platform']=='sentinel-2c'
    assert meta['provider_item_id']=='test-item'
    assert meta['valid_fraction']==1
    assert meta['source_pixel_size_crs_units']==[.001,.001]
    assert meta['reference_available'] is False
