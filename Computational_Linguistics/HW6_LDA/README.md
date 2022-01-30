- This homework is about a word alignment using LDA topic modeling and Gibbs sampling.
- HW6_LDA.ipynb is the file with report, 'solution.py' is a Python script which contains main implementaion code.
- 'data' directory contains:
    set of movie reviews (base dataset);
    custom corpus of news data in Belarusian and Russian;
- 'models' directory contains trained models with different hyperparams.
- Simply execute all cells, required data/packages will be uploaded automatically.
- This project can be runned via Google Colaboratory using the following [link](https://colab.research.google.com/github/tsimafeip/LCT-master-course/blob/main/Computational_Linguistics/HW6_LDA/HW6_LDA.ipynb). This notebook installs several additional programs and download data/scripts from git, so Colab session will save a local space of your computer.
- Training process with different sets of parameters (mainly NUM_OF_ITERATIONS) could result in comparably long training times (~3-4 hours). Observed times (local Apple M1 laptop) are illustrated in the report.

- EXTRA: 
    1) I applied this model to custom [Belarusian corpus] (!https://github.com/tsimafeip/Translator/tree/master/Data).
    I do not remove stopwords/punctuation and consider every sentence as document.
    2) I applied pyLDAvis librarys to analyse trained models.
    3) I added optional filtering for too rare/too common words.

- Conclusion:
    LDA topic modeling definitely learns something, since resulting topics have different PCA representation.
    However, I cannot say that I find topic semantically very coherent. I see several ways to explain it:
        1) Movie corpus is homogeneous, so possibly news data can show larger variety in topics
        2) Data preprocessing step can exclude verbs, do stemming, lemmatization, so we can reduce noise in data
        3) Hyperparameters tuning can drastically affect final distribution
