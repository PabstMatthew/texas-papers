import string
import os
import re
import sys

from difflib import SequenceMatcher
from urllib.request import urlretrieve
from urllib.parse import urlparse
import pytesseract
import nltk

sys.path.append('/home/matthewp/texas-papers')
from utils.utils import *

'''
    OCR configuration variables.
'''
ocr_config_dict = {
        'tessedit_char_whitelist': "'"+string.ascii_letters+string.digits+".?!“”’-—,;:$() '",
        'tessedit_fix_hyphens': '1',
        'enable_noise_removal': '1',
        'tessedit_fix_fuzzy_spaces': '1',
        'chs_leading_punct': "'\"'",
        'rej_use_sensible_wd': '1',
}
ocr_config_str = ' '.join(['-c '+key+'='+value for key, value in ocr_config_dict.items()])

'''
    Uses OCR and post-processing to convert an image to text.
        url: a URL pointing to an image with text in it.
        returns: a string of the post-processed text in the image.
'''
def img2txt(url):
    txt = ocr(url)
    txt = postprocess(txt)
    return txt

'''
    Uses OCR to convert an image to text.
        url: a URL pointing to an image with text in it.
        returns: a string of the text in the image.
'''
def ocr(url, cleanup=True):
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
        txt = pytesseract.image_to_string(fname, config=ocr_config_str)
        dbg_end()
    except pytesseract.pytesseract.TesseractNotFoundError:
        print('You need to install tesseract with your package manager!')

    # Cleanup and cache the result.
    if cleanup:
        os.remove(fname)
    if txt != '':
        cache_write(scope, fname, txt)
    return txt


'''
Methods to compute the similarity of two strings.
    word_similarity uses the number of word occurrences shared between strings.
    char_similarity uses a more generic string similarity metric.
        ground_truth: a string representing the "correct" string.
        comparison: the string to be compared to the correct string.
        returns: a value from 0.0 to 1.0 indicating the similarity.
'''
def word_similarity(ground_truth, comparison):
    truth = nltk.FreqDist(nltk.tokenize.word_tokenize(ground_truth))
    common = 0
    for word, freq in nltk.FreqDist(nltk.tokenize.word_tokenize(comparison)).items():
        if word in truth:
            truth_freq = truth[word]
            if freq <= truth_freq:
                common += freq
            else:
                common += max(0, truth_freq-freq)
    return common / truth.N()

def char_similarity(ground_truth, comparison):
    return SequenceMatcher(None, ground_truth, comparison).ratio()


'''
    Methods to postprocess text from an OCR'd image.
        txt: the text to be processed.
        returns: the processed text.
'''

# Concatenates word separated by hyphens at the end of lines.
def fix_hyphens(txt):
    def replace_hyphen(match_obj):
        if match_obj.group(1) is not None:
            return match_obj.group(1)
    return re.sub('([^\s])-\s+', replace_hyphen, txt)

# Combines a paragraph split by lines into a single line.
def simplify_paragraphs(txt):
    def simplify(match_obj):
        res = match_obj.group(1)+' '+match_obj.group(2)
        if res[-1] != ' ':
            res += ' '
        return res
    return re.sub('([^\n]+)\n([^\n]+)\n', simplify, txt)

# Replaces continuous whitespace with a space.
def consolidate_whitespace(txt):
    return re.sub('\s+', ' ', txt)

# Converts all-caps words to a normally capitalized word.
def allcaps2firstcaps(txt):
    def decaps(match_obj):
        return match_obj.group(1)+match_obj.group(2).lower()
    return re.sub('([A-Z])([A-Z]+)', decaps, txt)

# Removes random punctutation artefacts from OCR.
def remove_spurious_punctuation(txt):
    # Remove punctutation surrounded by spaces caused by random marks.
    txt = re.sub(r' [.\-’]+ ', ' ', txt)
    # Remove lines of em-dashes
    txt = re.sub(r'——+', '', txt)
    # Remove random punctuation before words.
    txt = re.sub(r' [\-—’]+\w', ' ', txt)
    return txt

# Combines all the above methods on OCR'd text.
def postprocess(txt):
    txt = fix_hyphens(txt)
    #txt = simplify_paragraphs(txt)
    txt = consolidate_whitespace(txt)
    txt = allcaps2firstcaps(txt)
    txt = remove_spurious_punctuation(txt)
    #txt = fix_spelling(txt)
    return txt

if __name__ == '__main__':
    test_url = 'https://chroniclingamerica.loc.gov/lccn/sn86088296/1883-04-19/ed-1/seq-1.jp2'
    txt = img2txt(test_url)
    end_idx = txt.find('Special Telegram')
    if end_idx != -1:
        txt = txt[:end_idx]
    else:
        warn('Failed to end text!')
        dbg(txt)
    with open('corpus/ground-truth-lccn_sn86088296_1883-04-19_ed-1_seq-1.txt', 'r') as f:
        ground_truth = f.read()
        ground_truth = simplify_paragraphs(ground_truth)
    #print(txt)
    #print(ground_truth)
    print(word_similarity(ground_truth, txt))

'''
    Code to do autocorrect, which didn't improve similarity scores.
'''

'''
import pkg_resources
from symspellpy import SymSpell, Verbosity

def fix_spelling(txt):
    # Dictionary initialization.
    max_edit = 1
    sym_spell = SymSpell(max_dictionary_edit_distance=max_edit, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt")
    bigram_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    def fix_word(match_obj):
        word = match_obj.group(0)
        # Ignore capitalized words.
        if word[0].isupper():
            return word
        # Ignore words that should be in the dictionary.
        whitelist = {'s', 'p', 'm', 'cirstance', 'pharmaceutist', 'imperiled', 'memoranda', 'distractive', 'favorably', 'brakemen', 'paralyzed', 'commissed', 'annualy', 'cremationists', 'labor', 'honored', 'fulfillment', 'saltpeter', 'jewelry', 'accidently', 'rumored', 'ocurred'}
        if word in whitelist:
            return word
        # First, try to split this word up.
        suggestions = sym_spell.lookup_compound(word, max_edit_distance=max_edit)
        if len(suggestions) > 0:
            suggestion = suggestions[0]
            if suggestion.distance > 0 and suggestion.distance <= max_edit and suggestion.count > 100:
                #dbg('{} -> {}'.format(word, suggestion))
                return suggestion.term
        # Then check for similar words.
        suggestions = sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=max_edit)
        if len(suggestions) > 0:
            suggestion = suggestions[0]
            if suggestion.distance > 0 and suggestion.distance <= max_edit and suggestion.count > 100:
                #dbg('{} -> {}'.format(word, suggestion))
                return suggestion.term
        return word

    txt = re.sub('[a-zA-Z]+', fix_word, txt)
    return txt
'''
