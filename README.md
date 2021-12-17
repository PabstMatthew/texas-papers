# Texas Papers
An analysis of the linguistic similarities between Texan cities' word usage in newspapers over time.

## Setting up
To setup this project locally, you can run the following commands. I'm assuming you have `python3` installed.
```
git clone https://github.com/pabstmatthew/texas-papers
cd texas-papers
pip3 install -r requirements.txt
mkdir resources
cd resources
gdown --id 1LKzjXkOamQO0Tr9yTUE1rOGvPvfPSqbD
unzip resources.zip
cd ..
```

## Building models
To train all models (which will be cached in a `.cache` directory), you can run this command: `python3 src/model.py`

To explore a particular model, you can also run the previous script with an argument specifying the name of the 
model that you're interested in, e.g. `python3 src/model.py Austin-Weekly-Statesman-1883-1898`

In this mode, the script will repeatedly ask you for words. If the word is present in the model, you will be 
shown nearest neighbors of the word, examples of the word's usage from the corpus, etc.

## Comparing models


## (Re-)Building all corpora
Building the database of corpora takes a significant amount of processing power and time, which is why I've 
included them pre-built in this repository. However, if you want to re-build them, or build your own set of 
corpora, I'm including some information here.

The set of corpora is defined in `src/corpus.py`, as a dictionary named `corpus_info`. The keys to `corpus_info` 
are the unique identifiers of the corpora, that are used to reference them throughout this project. The values 
in `corpus_info` contain information needed to scrape data for each corpus. The first element of the value is a 
unique identifier used by the Library of Congress to identify sets of resources. The second element is a range of 
years to scrape newspaper images from. These two elements are used in `src/sources.py` to scrape a list of image 
links for each corpus in `corpus_info`, using knowledge about the structure of URLs in the Library of Congress 
database. Once these links are collected, text is scraped from random images until the number of words reaches a 
threshold defined by the constant `TARGET_EXP` in `src/sources.py`.

To actually build the corpora defined in `corpus_info`, you can simply run `python src/corpus.py`.

