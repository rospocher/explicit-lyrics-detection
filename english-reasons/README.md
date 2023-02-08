# Assessing fine-grained explicitness of song lyrics (English Lyrics)

The classical problem of detecting explicit song lyrics can be formulated as follow: given the lyrics of a song, determine if it contains content that is inappropriate or hurtful for children. That is, a binary output (explicit / non-explicit) is foreseen.

In this work we propose to go beyond this mere binary formulation by requiring that, in case of explicitness, additional details on the reason(s) for the explicitness should  be provided. More in details, we propose to codify the following distinct reasons for the explicitness of song lyrics so that, given the lyrics of a song, if it is predicted as explicit, one or more of these reasons has also to be suggested:
* *Strong language*: the song lyrics includes offensive words or curse words, i.e., words generally found to be disturbing and that are not normally used in regular conversation
* *Substance abuse*: the song lyrics refer to excessive use (e.g., to get stoned, to get high, to indulge in a dependency) of a drug, alcohol, prescription medicine, etc., in a way that is detrimental to self, society, or both.
* *Sexual reference*: the song lyrics contain references to sex, sexual organs, sexual body parts, sexual activity, sexual abuse, and so on.
* *Reference to violence*: the song lyrics contain references to hurting a person or living being intentionally, including description or suggestion of acts typically considered as violent (e.g., killing, stabbing, mentally or physically torturing, committing suicide).
* *Discriminatory language*: the song lyrics contain (i) insulting or pejorative expressions referring to races, ethnic groups, nationalities, genders, sexual orientation, etc.; (ii) offensive language directed at one specific subset of people; (iii) reiteration of stereotypes that can be hurtful for a specific target of people.


## Code

The [code folder](code) contains the Python code for:

* training and testing a 1D-CNN classifier with FastText embeddings ([1dcnn.py](code/1dcnn.py))
* fine-tuning and evaluating a (Neural Language Model) NLM multi-label classifier ([LM_distilbert.py](code/lm_training.py)) build on top of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased). A pre-trained model for fine-grained explicit lyrics prediction can be found [here](https://drive.google.com/drive/folders/1T9tW0Ev3tGr9JG7I0JaEsSssuPYJiWU1?usp=sharing).


## Dataset

The [dataset folder](dataset) contains a novel dataset of 4,000 song lyrics manually annotated with explicitness information. Besides the indication of whether the song lyrics contain explicit content or not, each explicit song lyrics is also appropriately annotated according to the five reasons for explicitness previously mentioned. To the best of our knowledge, this is: (i) the first released dataset containing manually curated information on the explicitness of song lyrics; and (ii), the first dataset containing fine-grained explicitness annotations, detailing the reasons for the explicitness of song lyrics.

Due to licensing issues, besides the explicitness metadata information, we can only make available the LyricWiki page ID of each lyrics, from which the full text of the lyrics can be retrieved from the [Internet Archive dump](https://archive.org/download/wiki-lyricsfandomcom/).
The Spotify song ID is also provided.


## Publications

* **Assessing fine-grained explicitness of song lyrics**<br/>
    By Marco Rospocher and Samaneh Eksir<br/>
    Under review<br/>