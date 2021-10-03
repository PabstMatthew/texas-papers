import re
import sys
import time
import random
from urllib.request import urlopen

import nltk

sys.path.append('.')
from utils.utils import *
from corpus.ocr import img2txt

'''
    A set of newspaper that have high-quality scans.
'''
corpuses = {
        'Dallas-Daily-Herald':          ('sn83025733', range(1873, 1887)),
        'Austin-Weekly-Statesman':      ('sn86088296', range(1883, 1898)),
        'Waco-Evening-News':            ('sn86088201', range(1892, 1894)),
        'San-Marcos-Free-Press':        ('sn86088181', range(1877, 1890)),
        'San-Antonio-Light':            ('sn87090966', range(1883, 1886)),
        'Fort-Worth-Daily-Gazette':     ('sn86064205', range(1883, 1890)),
        'Brownsville-Daily-Herald':     ('sn86099906', range(1897, 1909)),
        'Bryan-Morning-Eagle':          ('sn86088652', range(1989, 1909)),
        'El-Paso-Daily-Herald':         ('sn86064199', range(1896, 1901)),
}

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
            time.sleep(1)
            warn('Encountered error: {}'.format(e))
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

def build_link_database():
    for name, data in corpuses.items():
        info('Building image link database for "{}"'.format(name))
        resource_id = data[0]
        years = data[1]
        scrape_image_links(name, resource_id, years)

'''
    Scrapes text from a list of links to form a corpus of text.
        links: a list of strings that are links to images.
        returns: a string containing all the text.
'''
def scrape_text(name, links):
    # Number of words to scrape for the corpus.
    TARGET_EXP = 6
    TARGET_CORPUS_SIZE = 10 ** TARGET_EXP
    scope = 'Text'
    # Check the cache first.
    name += '-10e' + str(TARGET_EXP)
    cached_txt = cache_read(scope, name)
    if cached_txt:
        return cached_txt
    # Process the links in a random order.
    num_words = 0
    txts = []
    random.shuffle(links)
    for link in links:
        info('Processing "{}" ...'.format(link))
        txt = img2txt(link)
        txts.append(txt)
        words = nltk.word_tokenize(txt)
        num_words += sum(map(lambda word: 1 if word.isalpha() else 0, words))
        completion = num_words / TARGET_CORPUS_SIZE
        info('  Finished! {:.2f}% complete with this corpus.'.format(completion*100.0))
        if completion >= 1.0:
            break
    info('Completed creating corpus "{}".'.format(name))
    cache_write(scope, name, links)
    return '\n'.join(txts)

def build_corpuses():
    for name, data in corpuses.items():
        info('Building corpus for "{}"'.format(name))
        resource_id = data[0]
        years = data[1]
        links = scrape_image_links(name, resource_id, years)
        corpus = scrape_text(name, links)

if __name__ == '__main__':
    build_corpuses()

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
