# Explicit Lyrics Detection (English Lyrics)

This repository collects materials related to the detection of explicit song lyrics, i.e., determining if the lyrics of a given song could be offensive or unsuitable for children.

## Code

The [code folder](code) contains the Python code for:

* training and testing a logistic regression classifier with Bag-of-Word embeddings ([lrBOW.py](code/lrBOW.py))
* training and testing a FastText classifier ([fasttext.py](code/fasttext.py))
* training and testing a 1D-CNN classifier with FastText embeddings ([1dcnn.py](code/1dcnn.py))
* fine-tuning (Neural Language Model) NLM classifiers ([lm_training.py](code/lm_training.py)). Considered Language Models (from HuggingFace):
  * [bert-base-uncased](https://huggingface.co/bert-base-uncased)
  * [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)
  * [roberta-base](https://huggingface.co/roberta-base)
  * [xlnet-base-cased](https://huggingface.co/xlnet-base-cased)
  * [microsoft/deberta-base](https://huggingface.co/microsoft/deberta-base)
* predicting with NLM classifiers ([lm_inferencing.py](code/lm_inferencing.py)) 
* evaluating the predicted labels ([evaluate.py](code/evaluate.py)) 
* computing statistical significance with the McNemar test ([mcnemar.py](code/mcnemar.py)) 


## Datasets, Predictions, Scores

The [data folder](data) contains links to the dataset (train and test splits), classifier predicitons, and classifier scores.

To build the dataset we relied on content provided through public platforms, namely LyricWiki and Spotify.
Due to licensing issues, besides the explicitness metadata information (from Spotify), we can only make available the LyricWiki page ID of each lyrics, from which the full text of the lyrics can be retrieved from the [Internet Archive dump](https://archive.org/download/wiki-lyricsfandomcom/).

## Publications

* **[Explicit song lyrics detection with subword-enriched word embeddings](https://doi.org/10.1016/j.eswa.2020.113749)**<br/>
    By Marco Rospocher<br/>
    In Expert Systems with Applications, Volume 163, January 2021, 113749<br/>
    [[bib](https://marcorospocher.com/files/bibs/2021eswa.bib)] 
    
* **[On exploiting Transformers for detecting explicit song lyrics](https://doi.org/10.1016/j.entcom.2022.100508)**<br/>
    By Marco Rospocher<br/>
    In Entertainment Computing, Volume 43, 2022, 100508,<br/>
    [[bib](https://marcorospocher.com/files/bibs/2022ec.bib)] 
