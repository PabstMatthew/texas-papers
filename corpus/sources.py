import re
import sys
import time
import random
from urllib.request import urlopen

import nltk

try:
    from corpus.ocr import img2txt
except ImportError:
    from ocr import img2txt
sys.path.append('.')
from utils.utils import *

'''
    Scrapes links to newspaper scans given a resource ID and set of years.
        resource_id: a string like 'sn83025733' identifying the particular newspaper.
        years: an iterable of years that have newspaper scans.
'''
def scrape_image_links(name, resource_id, years):
    scope = 'ImageLinks'
    cached_result = cache_read(scope, name)
    if cached_result:
        return cached_result

    def get_html(url):
        dbg('Scraping '+url)
        try:
            fp = urlopen(url)
        except Exception as e:
            warn('Encountered error: {}'.format(e))
            time.sleep(1)
            warn('Retrying ...')
            fp = urlopen(url)
        data = fp.read()
        html = data.decode('utf8')
        return html

    base = 'https://chroniclingamerica.loc.gov'

    # Get a list of links to every edition available.
    ed_links = set()
    for year in years:
        url = '{}/lccn/{}/issues/{}'.format(base, resource_id, str(year))
        html = get_html(url)
        ed_links.update(re.findall('/lccn/{}/\d+-\d+-\d+/ed-\d+/'.format(resource_id), html))
    ed_links = list(map(lambda l: base+l, ed_links))

    # Get a list of links to sequences for each edition.
    seq_links = set()
    for ed_link in ed_links:
        html = get_html(ed_link)
        seq_links.update(re.findall('/lccn/{}/\d+-\d+-\d+/ed-\d+/seq-\d+/'.format(resource_id), html))
    seq_links = list(map(lambda l: base+l, seq_links))

    # Check which sequences have images.
    links = []
    for seq_link in seq_links:
        html = get_html(seq_link)
        if not 'Missing Page: Not digitized, published' in html:
            links.append(seq_link[:-1]+'.jp2')
    cache_write(scope, name, links)
    return links

'''
    Scrapes text from a list of links to form a corpus of text.
        links: a list of strings that are links to images.
        returns: a string containing all the text.
'''
def scrape_text(name, links):
    # Number of words to scrape for the corpus.
    TARGET_EXP = 7
    TARGET_CORPUS_SIZE = 10 ** TARGET_EXP
    scope = 'Text'
    # Check the cache first.
    name += '-10e' + str(TARGET_EXP)
    cached_txt = cache_read(scope, name)
    if cached_txt:
        return cached_txt
    txts = []
    num_words = 0
    def handle_txt(txt):
        nonlocal txts, num_words
        info('Processing "{}" ...'.format(link))
        txts.append(txt)
        words = nltk.word_tokenize(txt)
        num_words += sum(map(lambda word: 1 if word.isalpha() else 0, words))
        completion = num_words / TARGET_CORPUS_SIZE
        info('  Finished! {:.2f}% complete with this corpus.'.format(completion*100.0))
        return completion >= 1.0
    # First, use any links that were cached.
    uncached_links = []
    for link in links:
        txt = img2txt(link, cached_only=True)
        if txt:
            if handle_txt(txt):
                break
        else:
            uncached_links.append(link)
    # Process the remaining links in a random order.
    random.shuffle(uncached_links)
    for link in uncached_links:
        txt = img2txt(link)
        if handle_txt(txt):
            break
    info('Completed creating corpus "{}".'.format(name))
    txt = '\n'.join(txts)
    cache_write(scope, name, txt)
    return txt

'''
Marshall:       1849-1869
    (1K issues)
Dallas:         1855-1885, 1919-1922
    (5K issues)
Austin:         1871-1898
    (1.2K issues)
Waco:           1874-1889, 1892-1894
    (3K issues)
San Saba:       1876-1892
    (0.4K issues)
San Marcos:     1877-1890
    (0.6K issues)
San Antonio:    1883-1886
    (0.8K issues)
Fort Worth:     1883-1896
    (4K issues)
Brownsville:    1892-1909, 1912-1937
    (13K issues)
Houston:        1893-1903
    (2K issues)
Shiner:         1893-1921
    (0.7K issues)
Bryan:          1895-1917, 1892-1913
    (6K issues)
El Paso:        1896-1901, 1910-1921
    (4.6K issues)
Lubbock:        1908-1922
    (0.4K issues)
Corpus Christi: 1911-1921
    (2K issues)
Amarillo:       1911-1922, 1907-1920, 1905-1908
    (1.7K issues)
'''
