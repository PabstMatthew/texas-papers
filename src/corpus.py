import sys

import nltk
from nltk.probability import FreqDist

import sources
from utils import *

# Make sure NLTK resources are downloaded.
nltk_resources = [('tokenizers/punkt', 'punkt'),
                  ('tagger/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
                  ('corpora/stopwords', 'stopwords')]
for path, resource in nltk_resources:
    try:
        nltk.data.find(path)
    except LookupError:
        info('nltk resource "{}" not found, downloading now ...'.format(resource))
        nltk.download(resource)

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
    Checks the cache for a particular corpus.
        name: the name of the corpus
        returns: a list of the sentences in a corpus if it exists, otherwise None.
'''
def corpus(name):
    cached_text = resource_read('Text', name+sources.TARGET_EXP_SUFFIX)
    return cached_text

'''
    Gets the source link where a quote originated from.
        name: the name of the corpus
        returns: a string containing a link to the resource, or None
'''
def get_source_link(name, quote):
    cached_sources = resource_read('TextSources', name)
    if cached_sources is None:
        warn('Failed to load TextSources! You need to unzip resources.zip.')
        return None
    cached_sources = cached_sources.splitlines()
    for i, line in enumerate(corpus(name).splitlines()):
        idx = line.find(quote)
        if idx != -1:
            return cached_sources[i]
    return None

'''
    Splits a corpus into sentences.
        name: the name of the corpus
        returns: a list of the sentences in a corpus if it exists, otherwise None.
'''
def corpus_sentences(name):
    txt = corpus(name)
    if txt is None:
        return None
    scope = 'CorpusSentences'
    cached_sents = cache_read(scope, name)
    if cached_sents:
        return cached_sents
    sents = nltk.sent_tokenize(txt)
    cache_write(scope, name, sents)
    return sents

'''
    Generator that yields pairs of the form (<corpus_name>, <corpus_text>)
'''
def corpora():
    for name, data in corpus_info.items():
        txt = corpus(name)
        if txt:
            yield name, txt

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
    corpus_txt = cache_read('Text', corpus_name+TARGET_EXP_SUFFIX)
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
    for name, data in corpus_info.items():
        info('Building corpus for "{}"'.format(name))
        resource_id = data[0]
        years = data[1]
        links = sources.scrape_image_links(name, resource_id, years)
        corpus = sources.scrape_text(name, links)

'''
    Iterates over known corpuses and scrapes links to images in order to build out the cache 
    containing all image links for the corpus.
'''
def build_link_cache():
    for name, data in corpus_info.items():
        info('Building image link database for "{}"'.format(name))
        resource_id = data[0]
        years = data[1]
        sources.scrape_image_links(name, resource_id, years)

'''
    If this script is called as the main program, build the corpus from the sources. 
    This will take a very long time!
'''
if __name__ == '__main__':
    build_corpus_cache()

