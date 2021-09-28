import sys
sys.path.append('/home/matthewp/texas-papers')

import pytesseract
from urllib.request import urlretrieve
import os
from utils.utils import *

def img2txt(url, cleanup=True):
    dbg_start('Downloading file')
    fname, _ = urlretrieve(url)
    dbg_end()
    try:
        dbg_start('Starting OCR')
        txt = pytesseract.image_to_string(fname)
        dbg_end('Done!')
    except pytesseract.pytesseract.TesseractNotFoundError:
        print('You need to install tesseract with your package manager!')
        txt = ''
    if cleanup:
        os.remove(fname)
    return txt

if __name__ == '__main__':
    test_url = 'https://chroniclingamerica.loc.gov/lccn/sn86088296/1883-04-19/ed-1/seq-1.jp2'
    print(img2txt(test_url))

