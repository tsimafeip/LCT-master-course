This exercise is about Common Sense Extraction from a collection of documents.
General task: extract tuples like <animal, eats, smth> for the specific animal.

I ended up with a simple rule-based solution looking for 'eat' lemma and all nouns after it.
I have also tried extractive QA BERT-based models, but their inference time is very long.

Requirements:
1. Group plural and single nouns ('grass' and 'grasses').
2. Allow only small noun phrases and nouns.
3. Avoid nonsensical ('what' or 'etc.') and general extractions ('food').
4. Count occurrences count and output list of tuples with frequency.

This lab is based on a setup code, provided by the tutors:
git clone https://github.com/phongnt570/akbc22-lab07.git
cd akbc22-lab07
