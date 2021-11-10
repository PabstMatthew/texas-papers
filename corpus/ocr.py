import string
import os
import re
import sys
import time
from urllib.request import urlretrieve
from urllib.parse import urlparse
from difflib import SequenceMatcher

import pytesseract
import nltk

sys.path.append('.')
from utils.utils import *

'''
    OCR configuration variables.
'''
punctuation = '.?!“”’-—,;:$()'
punctuation_regex = '[.?!“”’\-—,;:$()]'
ocr_config_dict = {
        'tessedit_char_whitelist': "'"+string.ascii_letters+string.digits+punctuation+" '",
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
        cached_only: if True, then returns None if the text was uncached.
        returns: a string of the post-processed text in the image.
'''
def img2txt(url, cached_only=False):
    # Check cache.
    scope = 'ImageText'
    fname = urlparse(url).path[1:].replace('/', '_')
    cached_txt = cache_read(scope, fname)
    if cached_txt:
        return cached_txt
    # Do the OCR and post-processing.
    txt = ocr(url, cached_only=cached_only)
    if txt is None:
        # If something went wrong, don't cache the result.
        return None
    txt = postprocess(txt)
    cache_write(scope, fname, txt)
    return txt

'''
    Uses OCR to convert an image to text.
        url: a URL pointing to an image with text in it.
        returns: a string of the text in the image.
'''
def ocr(url, cleanup=True, cached_only=False):
    scope = 'OCRText'
    fname = urlparse(url).path[1:].replace('/', '_')

    # Check the cache.
    cached_txt = cache_read(scope, fname)
    if cached_txt:
        return cached_txt
    elif cached_only:
        return None

    # Download the file.
    dbg_start('Downloading file "{}"'.format(fname))
    tries = 0
    while True:
        if tries >= 10:
            # Just give up on this link at some point.
            warn('Failed 10 times in a row retrieving "{}". Skipping this URL...'.format(url))
            return None
        tries += 1
        try:
            # Try to download the image.
            urlretrieve(url, fname)
            break
        except Exception as e:
            # If something goes wrong, just sleep and try again.
            warn('Encountered error: {}'.format(e))
            time.sleep(tries)
            warn('Retrying ...')
    dbg_end()

    # Run the OCR.
    try:
        dbg_start('Starting OCR')
        txt = pytesseract.image_to_string(fname, config=ocr_config_str)
        dbg_end()
    except pytesseract.pytesseract.TesseractNotFoundError:
        err('You need to install tesseract with your package manager!')

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
    return re.sub('([A-Z])([A-Z]+)', lambda match: match.group(1)+match.group(2).lower(), txt)

# Removes random punctutation artefacts from OCR.
def remove_spurious_punctuation(txt):
    # Remove punctutation surrounded by spaces caused by random marks.
    txt = re.sub(r' [.\-’]+ ', ' ', txt)
    # Remove lines of em-dashes
    txt = re.sub(r'——+', '', txt)
    # Remove random punctuation before words.
    txt = re.sub(r' [.\-—’]+\w', ' ', txt)
    # Consolidate punctuation.
    txt = re.sub(r'[.]+', '.', txt)
    txt = re.sub(r'[’]+', '’', txt)
    # Remove random punctuation within words.
    def remove_middle(match_obj):
        return match_obj.group(1)+match_obj.group(2)
    txt = re.sub(r'(\w)[?,;:$!](\w)', remove_middle, txt)
    # Remove punctuation immediately after a period.
    txt = re.sub(r'\.\s*[?\-—,!$’]\s*', '. ', txt)
    # Remove punctuation followed by other punctutation.
    txt = re.sub('({p})(\s*{p})+'.format(p=punctuation_regex), lambda match: match.group(1), txt)
    return txt

def split_words(txt):
    return re.sub(r'([a-z])([A-Z])', lambda match: match.group(1)+' '+match.group(2), txt)

'''
    Code to do autocorrect.
'''
import pkg_resources
from symspellpy import SymSpell, Verbosity

def nonsense_word(word):
    word = word.lower()
    if len(word) == 1:
        return not word in {'a', 'i'} 
    if len(word) < 4:
        return False
    vwl_set = {'a', 'e', 'i', 'o', 'u', 'y'}
    con_set = set(string.ascii_lowercase).difference(vwl_set)
    vwls = 0
    for i in range(len(word)):
        c = word[i]
        if c.isdigit():
            return True
        if c in vwl_set:
            vwls += 1
        if i > 1:
            # 3 of the same letter in a row is nonsense.
            l_c = word[i-1]
            ll_c = word[i-2]
            if c == l_c and l_c == ll_c:
                return True
            if c in con_set and c != 'r' and c != 's' and ((c == l_c and ll_c in con_set) or (c == ll_c and l_c in con_set)):
                return True
        if i > 0:
            # Some pairs of letters should never occur.
            bad_pairs = {'cs', 'cv', 'vv', 'vc'}
            pair = word[i-1:i]
            if pair in bad_pairs:
                return True
    vwl_freq = vwls / len(word)
    # Some vowel frequencies are impossible.
    return vwl_freq < 0.1 or vwl_freq > 0.9

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
        if word.isdigit():
            return word
        #if not word[0].isupper():
        # Ignore words that should be in the dictionary.
        whitelist = {'Weinstein', 'San', 'Mr', 'Dr', 'Ms', 'Mrs', 's', 'p', 'm', 'cirstance', 'pharmaceutist', 'imperiled', 'memoranda', 'distractive', 'favorably', 'brakemen', 'paralyzed', 'commissed', 'annualy', 'cremationists', 'labor', 'honored', 'fulfillment', 'saltpeter', 'jewelry', 'accidently', 'rumored'}
        if word in whitelist:
            return word
        # Check for similar words.
        suggestions = sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=max_edit, transfer_casing=True)
        if len(suggestions) > 0:
            suggestion = suggestions[0]
            if suggestion.distance != 0 and suggestion.distance <= max_edit and suggestion.count > 100:
                #dbg('{} -> {}'.format(word, suggestion))
                return suggestion.term
        # Try to split this word up.
        suggestions = sym_spell.lookup_compound(word, max_edit_distance=max_edit, transfer_casing=True)
        if len(suggestions) > 0:
            suggestion = suggestions[0]
            if suggestion.distance != 0 and suggestion.distance <= max_edit and suggestion.count > 100:
                #dbg('{} -> {}'.format(word, suggestion))
                return suggestion.term
        # Check if this word is plausibly real.
        if nonsense_word(word):
            return ''
        return word

    txt = re.sub('[\w():;]*[\w]', fix_word, txt)
    return txt

def remove_spurious_letters(txt):
    return re.sub('\W([a-zA-Z]) [a-zA-Z]\W', lambda match: match.group(1), txt)

# Combines all the above methods on OCR'd text.
def postprocess(txt):
    txt = fix_hyphens(txt)
    #txt = simplify_paragraphs(txt)
    txt = consolidate_whitespace(txt)
    txt = allcaps2firstcaps(txt)
    txt = split_words(txt)
    txt = fix_spelling(txt)
    txt = remove_spurious_punctuation(txt)
    txt = remove_spurious_letters(txt)
    return txt

if __name__ == '__main__':
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    else:
        test_url = 'https://chroniclingamerica.loc.gov/lccn/sn86088296/1883-04-19/ed-1/seq-1.jp2'
        with open('corpus/ground-truth-lccn_sn86088296_1883-04-19_ed-1_seq-1.txt', 'r') as f:
            ground_truth = f.read()
            ground_truth = simplify_paragraphs(ground_truth)
    txt = img2txt(test_url)
    end_idx = txt.find('Special Telegram')
    if end_idx != -1:
        txt = txt[:end_idx]
    else:
        warn('Failed to end text!')
        dbg(txt)
    #print(txt)
    #print(ground_truth)
    #print(word_similarity(ground_truth, txt))

