- This exercise is about entity typing, specifically Named Entity Recognition.
- Some code ('run.py', 'evaluate.py', 'run-evaluate.sh') and data ('train.tsv', 'train-test-groundtruth.tsv') files were provided by tutors.
- I did some preprocessing - types lemmatization and pos-counting. Results are stored in *.json files.
- Approach
  - My main approach is to detect patterns like 'is a' and then do sub-phrase matching based on known types from training.
  - Backup approaches include POS-based matching: learn common POS patterns for types and look for them.
  - Also, I have implemented simple but effective approach: return all sub-spans which were seen as training types as types.
- Unfortunately, I make no use of entity information, relying solely on provided text. 
Probably, usage of neural approaches can leverage that.
- Extra packages: tqdm, spacy (for POS-tags only), nltk.