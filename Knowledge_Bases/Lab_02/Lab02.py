import csv
import re
from collections import defaultdict

import requests

from bs4 import BeautifulSoup as bs
from pprint import pprint
from typing import Dict, List, Union, Tuple

ENGLISH_URL_POSTFIX = '&language=en'
ADDITIONAL_LINKS_TABLE_HEADER = "Additional Links"
RESPONSIBLE_INSTRUCTORS_TABLE_HEADER = "Responsible Instructors"
SINGLE_INSTRUCTOR_TABLE_HEADER = "Responsible Instructor"
RESPONSIBILITIES_HEADER = "Responsibilities"


def problem_1(name: str) -> List[Dict[str, Union[str, List[str]]]]:
    '''
    1. Extract attributes from html content.
    2. Store them in list of dict pairs(attribute:value)
    3. Clean attribute values (references, so on).
    '''
    url = 'https://how-i-met-your-mother.fandom.com/wiki/' + name.replace(' ', '_')
    sample_response = requests.get(url)

    soup = bs(sample_response.content, 'html.parser')
    infobox = soup.find('table', attrs={'class': "infobox character"})
    # soup.select("table[summary='Grunddaten zur Veranstaltung']")
    tables = soup.select('table')
    attr_names = infobox.find_all('div', attrs={'style': "text-align:left; font-size:.75em; font-style:italic;"})
    attr_values = infobox.find_all('div', attrs={'style': "text-align:right; font-size:1em;"})

    assert len(attr_values) == len(attr_names)

    attribute_name_to_value = {}

    for attr_name, attr_value in zip(attr_names, attr_values):
        # TODO: clean values (from reference, for instance)
        # TODO: process lists correctly
        attribute_name_to_value[attr_name.text] = attr_value.text

    return [{'attribute': attr_name, 'value': attr_value} for attr_name, attr_value in attribute_name_to_value.items()]


def problem_2_1() -> List[Dict[str, str]]:
    """As a result, it is very page-specific with a set of parsing heuristics."""

    def get_seed_page_urls():
        seed_urls = []
        for a in soup.find_all('a', href=True, attrs={'class': 'ueb', 'title': re.compile('öffnen')}):
            seed_urls.append(a['href'])

        # to filter out lower-level urls
        max_len_href = max(len(href) for href in seed_urls)
        seed_urls = [href for href in seed_urls if len(href) == max_len_href]

        return seed_urls

    base_url = """
    https://www.lsf.uni-saarland.de/qisserver/rds?state=wtree&search=1&trex=step&root120221=320944|310559|318658|311255
    """

    response = requests.get(base_url + ENGLISH_URL_POSTFIX)
    soup = bs(response.content, 'html.parser')

    seed_page_urls = get_seed_page_urls()

    course_name_to_link = []

    for seed_page_url in seed_page_urls:
        response = requests.get(seed_page_url + ENGLISH_URL_POSTFIX)
        soup = bs(response.content, 'html.parser')

        for a in soup.find_all('a', href=True, attrs={'title': re.compile('More information about')}):
            course_name_to_link.append({a.text: a['href']})

    return course_name_to_link


def parse_table(table, multidata_headers: List[str]) -> Dict[str, Union[str, List[str]]]:
    rows = table.find_all('tr')

    header_id_to_text = {}
    header_id_to_data = defaultdict(list)

    run_naive_reading = False
    headers_to_resolve_from_naive = []

    for row in rows:
        for table_header in row.find_all('th'):
            header_id = table_header['id']
            header_text = table_header.text.strip()
            if header_id in header_id_to_text and header_id_to_text[header_id] != header_text:
                headers_to_resolve_from_naive.extend([header_text, header_id_to_text[header_id]])
                # header_id_to_texts[header_id].add(header_text)
                # header_id_to_texts[header_id].add(header_id_to_text[header_id])
                print(f'Duplicate header id for table \'{table["summary"]}\': {header_id}!')
                run_naive_reading = True
            header_id_to_text[header_id] = header_text

        for data in row.find_all('td'):
            if len(data['headers']) > 1:
                run_naive_reading = True
                # raise RuntimeError('Cannot parse table with two table headers for one data column!')
            data_header_id = data['headers'][0]
            header_id_to_data[data_header_id].append(data.text.strip())

    header_to_data = {}

    if run_naive_reading:
        headers = []
        data = []
        for i, row in enumerate(rows):
            for table_header in row.find_all('th'):
                rowspan = int(table_header.get('rowspan', 1))
                header_text = table_header.text.strip()
                headers.extend([header_text] * rowspan)
            data.extend([data.text.strip() for data in row.find_all('td')])

        naive_header_to_data = dict(zip(headers, data))

        for header_text in headers_to_resolve_from_naive:
            header_to_data[header_text] = naive_header_to_data[header_text]

    # if header_id_to_texts:
    #     for header_id, header_texts in header_id_to_texts.items():
    #         header_data = header_id_to_data[header_id]
    #         assert len(header_data) == len(header_texts)
    #
    #         for header_text, data in zip(header_texts, header_data):
    #             header_to_data[header_text] = data

    for header_id, header_text in header_id_to_text.items():
        if header_text in headers_to_resolve_from_naive:
            continue

        header_data = header_id_to_data[header_id]
        if header_text not in multidata_headers:
            if len(header_id_to_data[header_id]) > 1 and header_text not in header_to_data:
                raise RuntimeError(
                    f'Unable to resolve duplicate header id for table \'{table["summary"]}\': {header_id}!'
                )

            header_data = header_id_to_data[header_id][0]
        header_to_data[header_text] = header_data

    return header_to_data


def problem_2_2(url: str) -> Dict[str, Union[str, List[str]]]:
    '''

    # return empty string if necessary of empty list


    0. Scrape data, parse it.
    1. Parse basic info infobox.

    document.querySelector("table[summary='kkkekek']")
    soup.select("table[summary='kkkekek']")
    Term	SoSe 2022
    https://www.w3schools.com/cssref/css_selectors.asp
    '''
    sample_url = """https://www.lsf.uni-saarland.de/qisserver/rds?state=verpublish&status=init&vmfile=no&publishid=137261&moduleCall=webInfo&publishConfFile=webInfo&publishSubDir=veranstaltung"""

    if not url:
        # TODO: fix their markup bug, where assignment and credits have the same id
        # Possibly, reimplement old version with dummy table reading
        url = sample_url

    # url = """https://www.lsf.uni-saarland.de/qisserver/rds?state=verpublish&status=init&vmfile=no&publishid=134460&moduleCall=webInfo&publishConfFile=webInfo&publishSubDir=veranstaltung"""
    # url = """https://www.lsf.uni-saarland.de/qisserver/rds?state=verpublish&status=init&vmfile=no&publishid=136315&moduleCall=webInfo&publishConfFile=webInfo&publishSubDir=veranstaltung"""
    multidata_headers = [ADDITIONAL_LINKS_TABLE_HEADER, RESPONSIBLE_INSTRUCTORS_TABLE_HEADER, RESPONSIBILITIES_HEADER]
    response = requests.get(url + ENGLISH_URL_POSTFIX)
    soup = bs(response.content, 'html.parser')

    basic_info_table = soup.select_one("table[summary='Grunddaten zur Veranstaltung']")
    attribute_name_to_value = parse_table(basic_info_table, multidata_headers)

    instructors_table = soup.select_one("table[summary='Verantwortliche Dozenten']")
    responsible_instructors_parsed = parse_table(instructors_table, multidata_headers)
    responsible_instructors = responsible_instructors_parsed.get(RESPONSIBLE_INSTRUCTORS_TABLE_HEADER)
    if not responsible_instructors:
        responsible_instructors = responsible_instructors_parsed[SINGLE_INSTRUCTOR_TABLE_HEADER]
    attribute_name_to_value[RESPONSIBLE_INSTRUCTORS_TABLE_HEADER] = responsible_instructors

    return attribute_name_to_value


def problem_2_3() -> None:
    """16 fields"""
    pass


def main():
    # You can call your functions here to test their behaviours.
    pprint(problem_1("Lily Aldrin"))
    pprint(problem_2_1())
    pprint(problem_2_2(""))
    pprint(problem_2_3())


if __name__ == "__main__":
    main()
