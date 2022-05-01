import os
import re
import json
from typing import Tuple, Dict, List

from spacy import Language
from tqdm import tqdm
from wiki_dump_reader import Cleaner
from collections import Counter, defaultdict

import spacy


def remove_urls(text: str) -> str:
    url_start_indices = [_.start() for _ in re.finditer('http', text)]
    ok_text_start = 0

    no_url_text = []

    for i, url_start_index in enumerate(url_start_indices):
        no_url_text.append(text[ok_text_start:url_start_index])

        url_end_index = url_start_index + 1
        while text[url_end_index] not in ['\n', " "]:
            url_end_index += 1
        ok_text_start = url_end_index + 1

    cleaned_text = "".join(no_url_text)
    return cleaned_text


def preprocess_text(text: str, enable_cleaner: bool) -> str:
    if enable_cleaner:
        '''
        I am not totally sure about correctness of this cleaning, since Cleaner is dropping a lot of wiki markups.
        On the other hand, I am not sure if wiki markups are named entities that we are looking for,
        since they are not a part of genuine article text.
        At the moment I have decided to omit it in general case and apply for Problem 2_2, since we need sentences there.
        '''
        cleaner = Cleaner()
        text = cleaner.clean_text(text)
    else:
        text = remove_urls(text)

    '''                
    I understand that cleaning text in this style (number of replaces) is not effective,
    since we do full scan for each single replace.
    However, it is not a production system, and I hope it's acceptable (time is still very low, <0.01s).
    
    All the rules here were created based on trial-and-error approach from interim output files.
    For instance, 'the' introduces separate named entities: 'the European Union' and 'European Union'.
    
    However, there is still an issue like 'U.S.' and 'United States' being two different entities.
    '''

    # It is still mostly valid for non-cleaner case, but I would like to keep it general.
    text = text.replace('url-status=dead', '').replace('url-status = dead', '') \
        .replace('last=', '').replace('first=', '') \
        .replace('{{cite', '').replace('{{Cite', '').replace('the ', '') \
        .replace('|', ' ').replace(':', ' ') \
        .replace('{{', '').replace('}}', '') \
        .replace('[[', '').replace(']]', '') \
        .replace('====', '').replace('===', '').replace('==', '').replace("''", "") \
        .replace('</ref', ' ').replace('<ref', ' ').replace('<br', ' ').replace('=', ' ').replace('-', ' ')

    return text


def preprocess_files(source_dirpath: str = "wikipedia_dump",
                     target_dirpath: str = 'cleaned_wiki_dump',
                     force: bool = False,
                     enable_cleaner: bool = False) -> str:
    if os.path.exists(target_dirpath) and not force:
        return target_dirpath
    elif not os.path.exists(target_dirpath):
        os.mkdir(target_dirpath)

    for _, _, filenames in os.walk(source_dirpath):
        for filename in tqdm(filenames):
            source_filepath = os.path.join(source_dirpath, filename)
            target_filepath = os.path.join(target_dirpath, filename)
            with open(source_filepath) as input_f, open(target_filepath, 'w') as target_f:
                text = input_f.read()
                text = preprocess_text(text, enable_cleaner)
                target_f.write(text)

    return target_dirpath


def solve_problem_1(nlp: Language, source_filedir: str, output_dirpath: str = "Problem1") -> Dict[str, List[str]]:
    if not os.path.exists(output_dirpath):
        os.mkdir(output_dirpath)

    article_title_to_named_entities = defaultdict(list)

    for _, _, filenames in os.walk(source_filedir):
        for filename in tqdm(filenames):
            source_filepath = os.path.join(source_filedir, filename)

            with open(source_filepath) as input_f:
                text = input_f.read()
                article_title = text.splitlines()[0]
                doc = nlp(text)

                entity_text_counter = Counter([entity.text.strip() for entity in doc.ents])
                article_title_to_named_entities[article_title] = list(entity_text_counter)

                with open(os.path.join(output_dirpath, article_title + '.csv'), 'w') as out_f:
                    out_f.write('Title, Named Entity, Frequency\n')

                    for entity_name, entity_count in entity_text_counter.most_common():
                        out_f.write(f'"{article_title}", "{entity_name}", "{entity_count}"\n')

    return article_title_to_named_entities


def solve_problem_2_1(nlp: Language, source_filedir: str, output_filename: str = 'Problem2_1.csv',
                      pos_to_detect: Tuple[str] = ('ADJ', 'VERB')):
    if os.path.exists(output_filename):
        os.remove(output_filename)

    with open(output_filename, 'a') as out_f:
        out_f.write('Title, POS Type, POS, Frequency\n')

        for _, _, filenames in os.walk(source_filedir):
            for input_filename in tqdm(filenames):
                filepath = os.path.join(source_filedir, input_filename)

                with open(filepath) as input_f:
                    text = input_f.read()
                    article_title = text.splitlines()[0]

                    doc = nlp(text)

                    for pos_name in pos_to_detect:
                        pos_counter = Counter(token.lemma_ for token in doc if token.pos_ == pos_name)

                        for token_lemma, token_count in pos_counter.most_common(n=5):
                            out_f.write(f'{article_title}, {pos_name}, {token_lemma}, {token_count}\n')


def alternative_solve_problem_2_2(nlp: Language, source_filedir: str, output_dirpath: str = "alt_Problem2_2"):
    if not os.path.exists(output_dirpath):
        os.mkdir(output_dirpath)

    for _, _, filenames in os.walk(source_filedir):
        for filename in tqdm(filenames):
            source_filepath = os.path.join(source_filedir, filename)

            with open(source_filepath) as input_f:
                text = input_f.read()
                article_title = text.splitlines()[0]

                doc = nlp(text)

                with open(os.path.join(output_dirpath, article_title + '.csv'), 'w') as output_f:
                    output_f.write('Title, Named Entity, Sentence\n')
                    for sentence in doc.sents:
                        for named_entity in sentence.ents:
                            output_f.write(f'{article_title}, {named_entity.text}, {sentence.text.strip()}\n')


def solve_problem_2_2(nlp: Language, source_filedir: str,
                      article_title_to_named_entities: Dict[str, List[str]],
                      output_dirpath: str = "Problem2_2", preprocess: bool = True):
    '''
    1. Read all files, preprocess on the fly with cleaning
    2. Split text to sentences.
    3. Find all entries and form output.
    '''
    if not os.path.exists(output_dirpath):
        os.mkdir(output_dirpath)

    for _, _, filenames in os.walk(source_filedir):
        for filename in tqdm(filenames):
            source_filepath = os.path.join(source_filedir, filename)

            with open(source_filepath) as input_f:
                text = input_f.read()
                article_title = text.splitlines()[0]
                if preprocess:
                    text = preprocess_text(text, enable_cleaner=True)

                doc = nlp(text)
                sentences = set()
                for sentence in doc.sents:
                    sentences.update(sentence.text.split('\n'))

                named_entities = article_title_to_named_entities[article_title]

                with open(os.path.join(output_dirpath, article_title + '.csv'), 'w') as output_f:
                    output_f.write('Title, Named Entity, Sentence\n')
                    for named_entity in named_entities:
                        for sentence in sentences:
                            if named_entity in sentence:
                                output_f.write(f'{article_title}, {named_entity}, {sentence.strip()}\n')


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    source_dirpath = "wikipedia_dump"
    processed_dirpath = preprocess_files(source_dirpath=source_dirpath, force=False)
    article_title_to_named_entities = solve_problem_1(nlp, processed_dirpath)

    # with open('result.json', 'w') as fp:
    #     json.dump(article_title_to_named_entities, fp)
    #
    # with open('result.json') as fp:
    #     article_title_to_named_entities = json.load(fp)
    solve_problem_2_1(nlp, processed_dirpath)

    # I use different preprocessing here (Cleaner) to omit non-sentence data.
    # Also, sentences extracted by spacy, but named entities are matched based on substr search.
    solve_problem_2_2(nlp, source_filedir=source_dirpath, preprocess=True,
                      article_title_to_named_entities=article_title_to_named_entities)

    # Here I just use native spacy means and default preprocessing, code is more elegant.
    # However, output is somewhat strange, since preprocessing was not ideal.
    alternative_solve_problem_2_2(nlp, source_filedir=processed_dirpath)
