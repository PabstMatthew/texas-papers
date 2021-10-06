import os

corpus_dictionary = []
dictionary_path = 'dictionary.txt'
if os.path.exists(dictionary_path):
    with open(dictionary_path, 'r') as f:
        for word in f.readlines():
            corpus_dictionary.append(word[:-1])
corpus_lookup = dict((word, i) for i, word in enumerate(corpus_dictionary))

