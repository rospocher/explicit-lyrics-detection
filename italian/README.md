# Explicit Lyrics Detection (Italian Lyrics)

This repository collects materials related to the detection of explicit song lyrics, i.e., determining if the lyrics of a given song could be offensive or unsuitable for children.

## Code

The code folder contains the Python code for:

* majority and dictionary based classifier baselines ([baselines.py](code/baselines.py)). Bad words file available at: https://github.com/napolux/paroleitaliane/blob/master/paroleitaliane/lista_badwords.txt
* training and testing a logistic regression classifier with Bag-of-Word embeddings ([lrBOW.py](code/lrBOW.py))
* training and testing a FastText classifier with FastText pre-trained Italian embeddings ([fasttext.py](code/fasttext.py))
* training and testing a 1D-CNN classifier with FastText pre-trained Italian embeddings ([1dcnn.py](code/1dcnn.py))
* fine-tuning and testing (Neural Language Model) NLM classifiers ([finetuning_nlm.py](code/finetuning_nlm.py)). Considered Language Models (from HuggingFace):
  * [Musixmatch/umberto-wikipedia-uncased-v1](https://huggingface.co/Musixmatch/umberto-wikipedia-uncased-v1)
  * [Musixmatch/umberto-commoncrawl-cased-v1](https://huggingface.co/Musixmatch/umberto-commoncrawl-cased-v1)
  * [bert-base-multilingual-uncased](https://huggingface.co/bert-base-multilingual-uncased)
  * [dbmdz/bert-base-italian-uncased](https://huggingface.co/dbmdz/bert-base-italian-uncased)
  * [m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alberto](https://huggingface.co/m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alberto)
  * [idb-ita/gilberto-uncased-from-camembert](https://huggingface.co/idb-ita/gilberto-uncased-from-camembert)

## Datasets, Predictions, Scores

The [data folder](data) contains links to the dataset (train and test splits), classifier predicitons, and classifier scores.

To build the dataset we relied on content provided through public platforms, namely LyricWiki and Spotify.
Due to licensing issues, besides the explicitness metadata information (from Spotify), we can only make available the LyricWiki page ID of each lyrics, from which the full text of the lyrics can be retrieved from the [Internet Archive dump](https://archive.org/download/wiki-lyricsfandomcom/).

## Publications

* Assessment on Italian Lyrics: Submitted, under review.
