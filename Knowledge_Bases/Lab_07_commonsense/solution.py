"""
Barebone code created by: Tuan-Phong Nguyen
Date: 2022-06-03
"""

import logging
from collections import Counter
from typing import Dict, List, Tuple, Any

import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
from transformers import pipeline

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

LOOK_AROUND_PATTERN_NAME = 'lookAroundPatterns'
LOOK_AFTER_PATTERN_NAME = 'lookAfterPatterns'
NONSENTIAL_WORDS = {'thousand', 'hundred', '?', 'sea', 'baby', 'day', '%', 'water', 'hour', 'time', 'kg'}
GENERAL_WORDS = {'food', 'animal', 'meal', 'source'}


def get_head_word(phrase: str) -> str:
    """
    Gets head word of a phrase.
    :param phrase: phrase
    :return: head word of a phrase
    """
    if phrase.strip() == "":
        return ""

    try:
        # try to get head word using SpaCy
        return list(nlp(phrase.lower().strip()).sents)[0].root.lemma_
    except Exception:
        # if error occurs, return the last word of the phrase
        return phrase.lower().strip().split()[-1]


def get_eating_patterns(animal: str) -> List[List[Dict[str, Any]]]:
    eat_pattern = [
        # {"LOWER": {"IN": [f"{animal}", f"{animal}s"]}},
        {"LEMMA": "eat"},
        {"POS": "NOUN"},
    ]

    patterns = [eat_pattern]

    return patterns


def get_token_matcher(words_to_look_around: List[str], words_to_look_after: List[str]) -> Matcher:
    token_matcher = Matcher(nlp.vocab)
    around_patterns = [
        [{"LEMMA": {"IN": words_to_look_around}}],
    ]
    token_matcher.add(LOOK_AROUND_PATTERN_NAME, around_patterns)

    after_patterns = [
        [{"LEMMA": {"IN": words_to_look_after}}],
    ]
    token_matcher.add(LOOK_AFTER_PATTERN_NAME, after_patterns)

    return token_matcher


def rule_based_solution(animal: str, documents: List[str]) -> List[str]:
    """Simple rule-based solution with a bag of empirically discovered patterns."""
    phrase_matcher = Matcher(nlp.vocab)
    phrase_matcher.add("strongEatPatterns", get_eating_patterns(animal=animal))

    words_to_look_around = ["diet", "food", "prey"]
    words_to_look_after = ["eat"]
    token_matcher = get_token_matcher(words_to_look_around=words_to_look_around,
                                      words_to_look_after=words_to_look_after)

    logger.info(f"Animal: \"{animal}\". Number of documents: {len(documents)}. Running SpaCy...")
    spacy_docs = nlp.pipe(texts=documents, n_process=1)

    words_to_exclude = set()
    words_to_exclude.update(NONSENTIAL_WORDS)
    words_to_exclude.update(GENERAL_WORDS)
    words_to_exclude.add(get_head_word(animal))

    diet_from_token_matching = Counter()
    diets = []
    # matcher-based approach gives a strong signal for precision
    for spacy_doc in spacy_docs:
        for match_id, start, end in phrase_matcher(spacy_doc):
            # string_id = nlp.vocab.strings[match_id]  # Get name of the pattern
            # span = spacy_doc[start:end]  # The matched span
            # print(match_id, string_id, start, end, span.text)
            # print(spacy_doc[end - 1].lemma_)
            diet_word = spacy_doc[end - 1].lemma_.lower()
            if diet_word not in words_to_exclude:
                diets.append(diet_word)

        for match_id, start, end in token_matcher(spacy_doc):
            span = spacy_doc[start:end]  # Matched span
            sent = span.sent  # Sentence containing matched span

            # look around pattern does not improve quality, so I do not update 'diets' collection here.
            if nlp.vocab.strings[match_id] == LOOK_AROUND_PATTERN_NAME:
                for token in sent:
                    if token.pos_ == "NOUN":
                        candidate_token = token.lemma_.lower()
                        if candidate_token not in words_to_exclude and candidate_token not in words_to_look_around:
                            diet_from_token_matching[candidate_token] += 1
            elif nlp.vocab.strings[match_id] == LOOK_AFTER_PATTERN_NAME:
                look_after = False
                for token in sent:
                    if token.lemma_ in words_to_look_after:
                        look_after = True
                        continue

                    if look_after and token.pos_ == "NOUN":
                        candidate_token = token.lemma_.lower()
                        if candidate_token not in words_to_exclude:
                            diets.append(candidate_token)
                            diet_from_token_matching[candidate_token] += 1

    return diets


def preprocess_pred_res(pred_res: str) -> List[str]:
    processed_diet = []
    for token in nlp(pred_res):
        cand_text = token.lemma_.lower()
        # alternative: nlp(pred_res).noun_chunks
        if token.pos_ == 'NOUN' and cand_text.isalpha():
            processed_diet.append(cand_text)

    return processed_diet


def bert_based_solution(animal: str, documents: List[str]):
    """BERT-based solution for extractive QA task.

    Things to try extra:
    1. Try different sizes of the context - currently, 500.
    2. Try different models
    """
    animal_head_word = get_head_word(animal)
    model_name = "deepset/xlm-roberta-large-squad2"
    # distilbert-base-cased-distilled-squad by default
    qa_model = pipeline("question-answering", model=model_name, tokenizer=model_name)
    question = f"What does {animal_head_word} eat?"
    diets = []
    chunk_size = 500
    for document in tqdm(documents):
        for chunk_start in range(0, len(document), chunk_size):
            prediction_res = qa_model(question=question, context=document[chunk_start:chunk_start + chunk_size])
            if prediction_res['score'] > 0.7:
                diets.append(preprocess_pred_res(prediction_res['answer']))

    return diets


def your_solution(animal: str, doc_list: List[Dict[str, str]]) -> List[Tuple[str, int]]:
    """
    Task: Extract things that the given animal eats. These things should be mentioned in the given list of documents.
    Each document in ``doc_list`` is a dictionary with keys ``animal``, ``url``, ``title`` and ``text``, whereas
    ``text`` points to the content of the document.

    :param animal: The animal to extract diets for.
    :param doc_list: A list of retrieved documents.
    :return: A list of things that the animal eats along with their frequencies.
    """

    logger.info(f"Animal: \"{animal}\". Number of documents: {len(doc_list)}.")

    # You can directly use the following list of documents, which is a list of str,
    # if you don't need other information (i.e., url, title).
    documents = [doc["text"] for doc in doc_list]

    # You must extract things that are explicitly mentioned in the documents.
    # You cannot use any external CSK resources (e.g., ConceptNet, Quasimodo, Ascent, etc.).

    diets = rule_based_solution(animal=animal, documents=documents)
    # diets = bert_based_solution(animal=animal, documents=documents)

    # Output example:
    return Counter(diets).most_common()
