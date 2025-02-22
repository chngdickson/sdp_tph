import base64
import io
import numpy as np
import codecs
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps

# Encoding to binary
def img_arr_to_b64(img_arr):
    """Grayscale"""
    img_pil = PIL.Image.fromarray(img_arr).convert('L')
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    data = f.getvalue()
    encData = codecs.encode(data, 'base64').decode()
    encData = encData.replace('\n', '')
    return encData

# Decoding Binary to  Numpy array
def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    f = io.BytesIO()
    f.write(img_data)
    img_pil = PIL.Image.open(f)
    #img_pil.show()
    img_arr = np.array(img_pil)
    return img_arr