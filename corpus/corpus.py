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
        'Dallas-Daily-Herald-1873-1879':        ('sn83025733', range(1873, 1880)),
        'Dallas-Daily-Herald-1880-1887':        ('sn83025733', range(1880, 1888)),
        'Austin-Weekly-Statesman-1883-1898':    ('sn86088296', range(1883, 1899)),
        'Waco-Evening-News-1892-1894':          ('sn86088201', range(1892, 1895)),
        'San-Marcos-Free-Press-1877-1890':      ('sn86088181', range(1877, 1891)),
        'San-Antonio-Light-1883-1886':          ('sn87090966', range(1883, 1887)),
        'Fort-Worth-Daily-Gazette-1883-1890':   ('sn86064205', range(1883, 1891)),
        'Brownsville-Daily-Herald-1897-1909':   ('sn86099906', range(1897, 1910)),
        'Bryan-Morning-Eagle-1898-1909':        ('sn86088652', range(1898, 1910)),
        'El-Paso-Daily-Herald-1896-1901':       ('sn86064199', range(1896, 1902)),
}

'''
    Generator that yields pairs of the form (<corpus_name>, <corpus_text>)
'''
def corpora():
    for name, data in corpus_info.items():
        cached_text = cache_read('Text', name+'-10e7')
        if cached_text:
            yield name, cached_text

'''
    Returns the frequency distribution of words in a corpus.
        corpus_name: the name of the corpus.
        returns: an nltk.FreqDist object representing the corpus' word distribution
'''
def corpus_word_distribution(corpus_name):
    scope = 'CorpusWordDistribution'
    cached_result = cache_read(scope, corpus_name)
    if cached_result:
        return cached_result
    corpus_txt = cache_read('Text', corpus_name+'-10e7')
    if corpus_txt is None:
        err('Corpus "{}" does not exist in the cache!'.format(corpus_name))
    words = nltk.word_tokenize(corpus_txt)
    dist = FreqDist(map(lambda word: word.lower(),
                    filter(lambda word: word.isalpha(), words)))
    cache_write(scope, corpus_name, dist)
    return dist

'''
    Iterates over known corpuses, scrapes image links, and scrapes text from all those images 
    in order to build out the cache containing all the corpus data.
'''
def build_corpus_cache():
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

