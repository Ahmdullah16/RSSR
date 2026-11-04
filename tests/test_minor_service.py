from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import io
import json
import zipfile
import numpy as np
from PIL import Image
import pytest
from minor_app import service as s


def test_catalog_has_independent_family_checkpoints():
    entries=s.catalog()
    assert set(entries)=={'Bicubic',*(f'M{n}' for n in range(1,8))}
    hashes=[spec['checkpoint']['sha256'] for key,spec in entries.items() if key!='Bicubic']
    assert len(set(hashes))==7


def test_preflight_rejects_resource_excess_without_resizing():
    image=np.zeros((513,513,3),np.uint8)
    assert not s.preflight(image,2)['allowed']
    with pytest.raises(ValueError): s.bicubic(image,2)
    assert image.shape==(513,513,3)
    for scale in s.SCALES:
        result=s.preflight(np.zeros((2,3,3),np.uint8),scale)
        assert result['output_width']==3*scale
        assert result['output_height']==2*scale


def test_image_formats_are_real_rgb_encodings():
    image=s.procedural_hr(64)
    for format in ('PNG','JPEG'):
        data=s.encode_image(image,format,91)
        with Image.open(io.BytesIO(data)) as reopened:
            assert reopened.format==format
            assert reopened.size==(64,64)
            assert reopened.mode=='RGB'
            if format=='PNG': np.testing.assert_array_equal(np.asarray(reopened),image)


def test_synthetic_shape_seed_and_parameter_effect():
    hr=s.procedural_hr(128)
    low,reference=s.synthetic_pair(hr,4,.5,5,17)
    assert low.shape==(32,32,3)
    np.testing.assert_array_equal(reference,hr)
    np.testing.assert_array_equal(low,s.synthetic_pair(hr,4,.5,5,17)[0])
    assert not np.array_equal(low,s.synthetic_pair(hr,4,.5,0,17)[0])
    with pytest.raises(ValueError): s.synthetic_pair(np.zeros((31,32,3),np.uint8),4)


def test_reference_requirements_and_metric_definitions():
    image=s.procedural_hr(32)
    assert 'psnr' not in s.evaluate(image)
    perfect=s.evaluate(image,image)
    assert perfect['psnr']==float('inf')
    assert perfect['ssim']==pytest.approx(1.)
    assert perfect['mse']==0
    with pytest.raises(ValueError): s.evaluate(image,image[:16])
    with pytest.raises(ValueError): s.evaluate(image,image,16)
    assert s.evaluate(image[:4,:4],image[:4,:4])['ssim'] is None


def test_bundle_has_valid_images_and_strict_metadata():
    result={'pixels':s.procedural_hr(32),'model_id':'M1','scale':2,'metadata':{'psnr':float('inf')}}
    with zipfile.ZipFile(io.BytesIO(s.result_bundle([result],'JPEG'))) as archive:
        assert archive.namelist()==['1-M1-x2.jpg','1-provenance.json']
        assert json.loads(archive.read('1-provenance.json'))['psnr']=='infinity'
        with Image.open(io.BytesIO(archive.read('1-M1-x2.jpg'))) as image: assert image.format=='JPEG'


def test_upload_decoding_accepts_rgb_and_checks_actual_file_contents():
    from rssr.uploads import decode_image
    image=s.procedural_hr(32)
    for fmt in ('PNG','JPEG'):
        decoded=decode_image(s.encode_image(image,fmt))
        assert decoded.shape==image.shape and decoded.dtype==np.uint8
    with pytest.raises(ValueError): decode_image(b'not an image')
    with pytest.raises(ValueError): decode_image(b'x'*(8*1024*1024+1))
    buf=io.BytesIO();Image.new('RGBA',(16,16)).save(buf,format='PNG')
    with pytest.raises(ValueError): decode_image(buf.getvalue())
