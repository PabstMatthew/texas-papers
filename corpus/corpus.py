import sys
sys.path.append('/home/matthewp/texas-papers')

import pytesseract
from urllib.request import urlretrieve
from urllib.parse import urlparse
import os
from utils.utils import *

def img2txt(url, cleanup=True):
    scope = 'OCRText'
    fname = urlparse(url).path[1:].replace('/', '_')

    # Check the cache.
    cached_txt = cache_read(scope, fname)
    if cached_txt:
        return cached_txt

    # Download the file.
    dbg_start('Downloading file "{}"'.format(fname))
    urlretrieve(url, fname)
    dbg_end()

    # Run the OCR.
    try:
        dbg_start('Starting OCR')
        txt = pytesseract.image_to_string(fname)
        dbg_end()
    except pytesseract.pytesseract.TesseractNotFoundError:
        print('You need to install tesseract with your package manager!')

    # Cleanup and cache the result.
    if cleanup:
        os.remove(fname)
    if txt != '':
        cache_write(scope, fname, txt)
    return txt

if __name__ == '__main__':
    test_url = 'https://chroniclingamerica.loc.gov/lccn/sn86088296/1883-04-19/ed-1/seq-1.jp2'
    print(img2txt(test_url))

