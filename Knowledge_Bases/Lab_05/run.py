'''
Created on Nov 25, 2019
Sample structure of run file run.py

@author: cxchu
@editor: ghoshs
'''

import sys
import csv


def your_extracting_function(input_file, result_file):
    '''
    This function reads the input file (e.g. input.csv)
    and extracts the required information of all given entity mentions.
    The results is saved in the result file (e.g. results.csv)
    '''
    with open(result_file, 'w', encoding='utf8') as fout:
        headers = ['entity', 'dateOfBirth', 'nationality', 'almaMater', 'awards', 'workPlaces']
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)

        with open(input_file, 'r', encoding='utf8') as fin:
            reader = csv.reader(fin)

            # skipping header row
            next(reader)

            for row in reader:
                entity = row[0]
                abstract = row[1]
                dateOfBirth, nationality, almaMater, awards, workPlace = [], [], [], [], []

                '''
                baseline: adding a random value 
                comment this out or remove this baseline 
                '''
                dateOfBirth.append('1961-1-1')
                nationality.append('United States')
                almaMater.append('Johns Hopkins University')
                awards.append('Nobel Prize in Physics')
                workPlace.append('Johns Hopkins University')

                '''
                extracting information 
                '''

                dateOfBirth += extract_dob(entity, abstract)
                nationality += extract_nationality(entity, abstract)
                almaMater += extract_almamater(entity, abstract)
                awards += extract_awards(entity, abstract)
                workPlace += extract_workpace(entity, abstract)

                writer.writerow(
                    [entity, str(dateOfBirth), str(nationality), str(almaMater), str(awards), str(workPlace)])


def extract_dob(entity, abstract, **kwargs):
    '''
    date of birth extraction funtion
    '''
    dob = []
    '''
    === your code goes here ===
    '''
    return dob


def extract_nationality(entity, abstract, **kwargs):
    '''
    nationality extraction function
    '''
    nationality = []
    '''
    === your code goes here ===
    '''
    return nationality


def extract_almamater(entity, abstract, **kwargs):
    '''
    alma mater extraction function
    '''
    almaMater = []
    '''
    === your code goes here ===
    '''
    return almaMater


def extract_awards(entity, abstract, **kwargs):
    '''
    awards extraction function
    '''
    awards = []
    '''
    === your code goes here ===
    '''
    return awards


def extract_workpace(entity, abstract, **kwargs):
    '''
    workplace extraction function
    '''
    workPlace = []
    '''
    === your code goes here ===
    '''
    return workPlace


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Expected exactly 2 argument: input file and result file')
    your_extracting_function(sys.argv[1], sys.argv[2])
