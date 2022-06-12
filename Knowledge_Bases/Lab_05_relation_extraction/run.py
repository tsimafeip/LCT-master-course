'''
Created on Nov 25, 2019
Sample structure of run file run.py

@author: cxchu
@editor: ghoshs
@student: tsimafeip
'''
import os.path
import sys
import csv
from typing import Optional, Dict, Set, Union, List

import spacy
from spacy.tokens import Span
import wget
from tqdm import tqdm
from dateutil import parser


def your_extracting_function(input_file, result_file):
    '''
    This function reads the input file (e.g. input.csv)
    and extracts the required information of all given entity mentions.
    The results is saved in the result file (e.g. results.csv)
    '''

    nlp = spacy.load('en_core_web_sm')
    nationality_to_country = get_nationality_to_country()
    counter = [0]

    with open(result_file, 'w', encoding='utf8') as fout:
        headers = ['entity', 'dateOfBirth', 'nationality', 'almaMater', 'awards', 'workPlaces']
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)

        with open(input_file, 'r', encoding='utf8') as fin:
            reader = csv.reader(fin)

            # skipping header row
            next(reader)

            for row in tqdm(reader):
                entity = row[0]
                abstract = row[1]
                dateOfBirth, nationality, almaMater, awards, workPlace = [], [], [], [], []

                '''
                baseline: adding a random value 
                comment this out or remove this baseline 
                '''
                # dateOfBirth.append('1961-1-1')
                # nationality.append('United States')
                # almaMater.append('Johns Hopkins University')
                # awards.append('Nobel Prize in Physics')
                # workPlace.append('Johns Hopkins University')

                '''
                extracting information 
                '''

                parsed_abstract = nlp(abstract)

                dateOfBirth += extract_dob(entity, abstract, parsed_abstract=parsed_abstract)
                nationality += extract_nationality(entity, abstract,
                                                   parsed_abstract=parsed_abstract,
                                                   nationality_to_country=nationality_to_country)
                almaMater += extract_almamater(entity, abstract, parsed_abstract=parsed_abstract,
                                               counter=counter)
                awards += extract_awards(entity, abstract, parsed_abstract=parsed_abstract)
                workPlace += extract_workpace(entity, abstract, parsed_abstract=parsed_abstract)

                writer.writerow(
                    [entity, str(dateOfBirth), str(nationality), str(almaMater), str(awards), str(workPlace)])

    print(counter)


def extract_first_date_from_sentence(sentence) -> Optional[str]:
    for named_entity in sentence.ents:
        if named_entity.label_ == "DATE":
            # Wikipedia sometimes contains non-traditional dash in dates (UTF-code 8211 vs. traditional 45)
            date = named_entity.text.replace('–', '-').split('-')[0].strip()
            return date


def extract_dob(entity, abstract, **kwargs):
    '''
    date of birth extraction function

    Ideas:
    1. Easy pattern - extract the first NE-detected date from the first sentence.
    2. Extract the first NE-detected from any further sentence with token 'born'.

    TODO: make 'born' pattern more strict
    - Word window after it ~5 words
    - Extract dates by regex, not only by NER
    '''

    dob = []

    # TODO: fix problem with Emanuel Derman.
    # Spacy does not recognize date here for some reason
    # Emanuel Derman (born c. 1945) is a Jewish South African-born academic, businessman and writer.

    parsed_abstract = kwargs['parsed_abstract']
    # counter = kwargs['counter']

    for i, sentence in enumerate(parsed_abstract.sents):
        if i == 0:
            # Date of birth is usually the first appearing date in the first sentence.
            date = extract_first_date_from_sentence(sentence)
            if date:
                dob.append(date)
                break
        else:
            # For other cases (mostly wrong formatting or parsing) we strictly look for the word 'born'.
            tokens = {token.text.lower() for token in sentence}
            if 'born' in tokens:
                date = extract_first_date_from_sentence(sentence=sentence)
                if date:
                    dob.append(date)
                    break

    # if not dob:
    #     counter[0] += 1
    #     #print(abstract, end='\n\n')

    dob = [parser.parse(date).strftime("%Y-%m-%d") for date in dob]

    return dob


def get_nationality_to_country() -> Dict[str, str]:
    git_path = 'https://raw.githubusercontent.com/knowitall/chunkedextractor/master/src/main/resources/edu/knowitall/chunkedextractor/demonyms.csv'
    local_path = 'denonyms.csv'
    if not os.path.isfile(local_path):
        wget.download(git_path, local_path)

    nationality_to_country = dict()
    with open(local_path, 'r', encoding='utf8') as fin:
        reader = csv.reader(fin)

        # skipping header row
        next(reader)

        for row in reader:
            nationality = row[0]
            country = row[1]
            nationality_to_country[nationality] = country

    return nationality_to_country


def extract_nationality(entity, abstract, **kwargs):
    '''
    nationality extraction function

    Ideas:
    1. First sentence usually state smth like American, Indian, Russian.
    We can extract nationality from here using adj-to-country static external mapping.
    2. Map birthplace to nationality: London -> United Kingdom.

    '''
    parsed_abstract = kwargs['parsed_abstract']
    nationality_to_country = kwargs['nationality_to_country']

    nationalities = {
        nationality_to_country.get(ent.text)
        for ent in next(parsed_abstract.sents).ents if ent.label_ == "NORP" and ent.text in nationality_to_country
    }

    return nationalities


def get_education_general_words() -> Set[str]:
    return {'degree', 'PhD', 'graduate', 'Ph.D.', 'B.A.', 'B.S.', 'bachelor',
            'master', 'M.A.', 'M.S.', 'educate', 'study', 'complete', 'M.Sc.'}


def get_education_org_words() -> Set[str]:
    return {'university', 'college', 'institute', 'school'}


def extract_almamater(entity, abstract, **kwargs):
    '''
    alma mater extraction function

    Ideas
    1. Go sentence-by-sentence and look for the following tokens: degree, PhD, graduate,
     Ph.D., B.A. (Jerome Seymour Bruner ), M.S., B.S. Bachelor, Master, educate, study, complete.

    Then, choose named entities with University, Institute, Center, College in them.
    '''

    almaMater = set()

    education_words = get_education_general_words()
    education_organizations = get_education_org_words()
    filter_out_following_forms = {'Study', 'Studies', 'studies'}

    parsed_abstract = kwargs['parsed_abstract']

    for sentence in parsed_abstract.sents:
        tokens = {token.lemma_.lower() for token in sentence if token.text not in filter_out_following_forms}
        if tokens.intersection(education_words):
            for entity in sentence.ents:
                # we can have an additional check about ORG label, but word matching is stronger
                # if entity.label_ == "ORG":
                entity_words = {token.text.lower() for token in entity}
                if entity_words.intersection(education_organizations):
                    if entity.label_ != "ORG":
                        entity_text = " ".join([token.text for token in entity])
                        entity = entity_text.split(' at ')[-1]

                    almaMater.add(drop_the_from_beginning(entity=entity))

    return list(almaMater)


def drop_the_from_beginning(entity: Union[Span, List[str]]):
    entity_text: List[str] = [token.text for token in entity] if isinstance(entity, Span) else entity

    # spacy often extract named entities with 'the' word in the beginning
    if entity_text[0].lower() == 'the':
        entity_text = entity_text[1:]

    return " ".join(entity_text)


def extract_awards(entity, abstract, **kwargs):
    '''
    awards extraction function

    Ideas
    1. Match named entities based on prize/award/medal words.

    TODO:
    1. Go from named entities to just text spans, since NER system is not perfect.
    2. Check overlapping awards and choose the biggest span.
    '''
    awards = []

    prize_words = {'prize', 'award', 'medal'}
    parsed_abstract = kwargs['parsed_abstract']

    for named_entity in parsed_abstract.ents:
        ent_tokens = {token.text.lower() for token in named_entity}
        if ent_tokens.intersection(prize_words):
            # spacy often extract awards with 'the' word in the beginning
            awards.append(drop_the_from_beginning(entity=named_entity))

    return awards


def get_work_words() -> Set[str]:
    return {'work', 'position', 'professor', 'director',
            'president', 'reader', 'editor', 'rector', 'serve', 'employ'}


def get_work_org_words() -> Set[str]:
    work_org_words = get_education_org_words()
    work_org_words.update({'center', 'laboratory'})
    return work_org_words


def extract_workpace(entity, abstract, **kwargs):
    '''
    workplace extraction function

    Ideas
    1. look for the following words:
    work, position, professor, director, president, reader, editor, rector, serve, employ.
    Then, match existing organisation from the sentence.
    '''
    workPlace = set()

    parsed_abstract = kwargs['parsed_abstract']
    work_words = get_work_words()
    work_organisations = get_work_org_words()

    for sentence in parsed_abstract.sents:
        tokens = {token.lemma_.lower() for token in sentence}
        if tokens.intersection(work_words):
            for entity in sentence.ents:
                entity_words = {token.text.lower() for token in entity}
                if entity_words.intersection(work_organisations):
                    workPlace.add(drop_the_from_beginning(entity=entity))

    return list(workPlace)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Expected exactly 2 argument: input file and result file')
    your_extracting_function(sys.argv[1], sys.argv[2])
