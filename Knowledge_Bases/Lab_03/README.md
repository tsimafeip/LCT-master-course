This exercise is about entity typing, specifically Named Entity Recognition.
All files from the 'setup' folder were provided by tutors.

Extra packages: tqdm, spacy (for POS-tags only).


SCORES:
1) Basic with fallback:
7676 - discovered
6729 - failed
5595 - no patterns
a) TRAIN: bash ./run_evaluate.sh train-test.tsv results.tsv train-test-groundtruth.tsv
Strict: Using exact matching:
        Macro Precision, Recall and F1: 0.24591237234987134     0.28787891233766233     0.26524594279914254
        Micro Precision, Recall and F1: 0.20119332430173234     0.282031597236252       0.23485077891053463
Loose: Using exact matching on the lemma of the head-word of the type:
        Macro Precision, Recall and F1: 0.301973340869654       0.3543946482683977      0.32609066162556355
        Micro Precision, Recall and F1: 0.24544353528490828     0.34797721752240557     0.2878522804039424
B) TEST: bash ./run_evaluate.sh test.tsv results.tsv test-groundtruth.tsv
Strict: Using exact matching:
        Macro Precision, Recall and F1: 0.2103816017316016      0.2370198412698412      0.22290770237126104
        Micro Precision, Recall and F1: 0.17089337175792507     0.23796147672552168     0.19892653471989266
Loose: Using exact matching on the lemma of the head-word of the type:
        Macro Precision, Recall and F1: 0.2679131493506492      0.3033505952380952      0.2845327192473591
        Micro Precision, Recall and F1: 0.21506646971935006     0.30282861896838603     0.25151148730350664
c) NEW-TRAIN - in case of discovered pattern:
2) Strict: Using exact matching:
        Macro Precision, Recall and F1: 0.4457073996873372      0.41428695501129104     0.4294231943119566
        Micro Precision, Recall and F1: 0.4374925798409118      0.37993607588411177     0.40668800353161905
Loose: Using exact matching on the lemma of the head-word of the type:
        Macro Precision, Recall and F1: 0.5282970731283655      0.5016022419911169      0.5146036946102642
        Micro Precision, Recall and F1: 0.5175240584531305      0.46954834537027057     0.4923702950152594
 
You can see quite high scores, so this approach works to some extent, 
but we need to address failed attempts and no-patterns.


NEXT STEPS
1) Try train neural networks: lstm/glove as for POS-prediction (large class size)
2) span predictor ?

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