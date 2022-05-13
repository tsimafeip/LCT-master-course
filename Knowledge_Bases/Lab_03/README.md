This exercise is about entity typing, specifically Named Entity Recognition.
All files from the 'setup' folder were provided by tutors.

Extra packages: tqdm, spacy (for POS-tags only), 


DONE:
1) Types in train and test file, how many new types do we have in test?
Answer: around 76 percent, so it makes sense to do inference after training.

2) Entity presence in sentence - is it the case every time?
Answer: yes, almost every time (~97% without first letter lowercasing, 99% with)
3) Class presence in sentence - is it the case every time?
Answer: unfortunately, not. Percentage of types present in sentence is around 32%.

TODO:
1) Read about Hearst patterns - start with simple (is-a) as a default implementation to have smth to submit.
We can later use it as a backup prediction for classes which are not appearing in the sentence.

a) Find POS tags - spacy
b) Apply rules
- find is-a
- do pos tags parsing
- pattern: [smth] is a(an)
{ 'NOUN', 'PROPN', 'ADJ NOUN', 'PROPN NOUN', 'NOUN ADP PROPN', 'NOUN NOUN', 'NOUN NOUN NOUN'}
- extend to such_as

Problem: no use of named_entity at all


2) Tweak a bit my lab about POS-prediction for NER-task




Questions:
1) Can I use ELMO?
2) There are extra named entities in the sentences, but I do not mark them.


Ideas:
1) Rule-based: use all this stuff from is-a.


2) NN-based: BERT embeddings + POS-tags + lemmas + trained classifier on top of it
ELMO(or BERT)+POS+LSTM