In this lab, we are working on **relation extraction**. Our end goal is to be able to
extract the following properties for given *entity* and their *Wikipedia abstract*: 
  - Date of Birth
  - Nationality
  - Alma Mater
  - Awards
  - Places of Work

In this exercise, we focus on using **pattern-based extraction**. 

Hints:<br>
1. You can use any tool to pre-process the data, 
like POS tagging or entity recognition.
2. You can use any other resources to improve your patterns, 
like dictionaries of relational paraphrases (e.g. RELLY or POLY1). 
3. You may also use pretrained word embeddings like word2vec or BERT. 
However, you are not allowed to look up relations in existing KBs like DBpedia, Wikidata, YAGO. 

Output format<br>
The properties must be extracted from the provided input text.
Your task is to complete the five property extraction functions called from your extracting function() inside run.py.
The output is saved in the results.csv file. 
Store entities as strings and properties as lists. 
For properties which have no value, an empty list is stored [].