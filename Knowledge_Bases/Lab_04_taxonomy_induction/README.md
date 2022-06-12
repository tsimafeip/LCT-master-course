General notes:
- This exercise is about taxonomy graph construction.
- Please rerun graph visualisation in case of bad picture.
- 'entity' node is always a root.
- Input data from WEBISALOD available for downloading [here](!https://www.mpi-inf.mpg.de/fileadmin/inf/d5/teaching/ss22_akbc/lab04_material.zip).
- I do some data preprocessing. It takes around 2-3 minutes to complete for the first run.
- I do not guarantee max depth 4, as requested in the task, but for sample data this rule holds.
- TaxonomyBuilder makes non-optimal decisions from time to time, but, in general, produces acceptable trees.

Algorithm details:
- Preprocess webisalod: replace + and _ with spaces, build hypernym-hyponym noisy graph (implemented as dict).
- There are two main sources of hypernym candidates: 
  - hypernym scores for specific hyponym
  - maximum word count, for words from all hypernyms (split by spaces). This approach tends to produce more general predictions.
- I sort all words from sample set by hyponyms count, since I expect less generic entities to have less hyponyms.
- Next, I generate all possible hypernyms from two sets and intersect them with sample words.
- Less generic words are linked only to more generic with a simple validation of edge quality.
- Later, I try to link words from sample set. Sometimes, I add intermediate nodes for taxonomy graph based on simple heuristics.

