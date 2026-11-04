"""Bounded opaque display-image decoding; independent of the web framework."""
import io
import numpy as np
from PIL import Image,ImageOps,UnidentifiedImageError


def decode_image(data):
    if len(data)>8*1024*1024:raise ValueError('Image file must be at most 8 MB')
    try:
        with Image.open(io.BytesIO(data)) as image:
            if max(image.size)>1024 or min(image.size)<1:
                raise ValueError('Choose an image at most 1024 pixels per side')
            if image.format not in ('PNG','JPEG'):
                raise ValueError('Choose a PNG or JPEG image')
            if image.mode not in ('RGB','L'):
                raise ValueError('Use opaque RGB or grayscale PNG/JPEG; remove transparency first')
            return np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as error:
        raise ValueError('The file is not a readable PNG or JPEG image') from error
