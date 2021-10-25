import sys

import nltk
from nltk.probability import FreqDist

try:
    from corpus.sources import scrape_image_links, scrape_text
except ImportError:
    from sources import scrape_image_links, scrape_text
sys.path.append('.')
from utils.utils import *

'''
    A set of newspaper that have high-quality scans.
'''
corpus_info = {
        'Dallas-Daily-Herald-1873-1887':        ('sn83025733', range(1873, 1887)),
        'Austin-Weekly-Statesman-1883-1898':    ('sn86088296', range(1883, 1898)),
        'Waco-Evening-News-1892-1894':          ('sn86088201', range(1892, 1894)),
        'San-Marcos-Free-Press-1877-1890':      ('sn86088181', range(1877, 1890)),
        'San-Antonio-Light-1883-1886':          ('sn87090966', range(1883, 1886)),
        'Fort-Worth-Daily-Gazette-1883-1890':   ('sn86064205', range(1883, 1890)),
        'Brownsville-Daily-Herald-1897-1909':   ('sn86099906', range(1897, 1909)),
        'Bryan-Morning-Eagle-1889-1899':        ('sn86088652', range(1889, 1899)),
        'Bryan-Morning-Eagle-1899-1909':        ('sn86088652', range(1900, 1909)),
        'El-Paso-Daily-Herald-1896-1901':       ('sn86064199', range(1896, 1901)),
}

'''
    Generator that yields pairs of the form (<corpus_name>, <corpus_text>)
'''
def corpora():
    for name, data in corpus_info.items():
        cached_text = cache_read('Text', name+'-10e6')
        if cached_text:
            yield name, cached_text

'''
    Iterates over known corpuses, scrapes image links, and scrapes text from all those images 
    in order to build out the cache containing all the corpus data.
'''
def build_corpus_cache():
    dictionary = None
    freq_threshold = 20
    for name, data in corpus_info.items():
        info('Building corpus for "{}"'.format(name))
        resource_id = data[0]
        years = data[1]
        links = scrape_image_links(name, resource_id, years)
        corpus = scrape_text(name, links)

'''
    Iterates over known corpuses and scrapes links to images in order to build out the cache 
    containing all image links for the corpus.
'''
def build_link_cache():
    for name, data in corpus_info.items():
        info('Building image link database for "{}"'.format(name))
        resource_id = data[0]
        years = data[1]
        scrape_image_links(name, resource_id, years)

if __name__ == '__main__':
    build_corpus_cache()

