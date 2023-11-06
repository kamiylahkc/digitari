import os
import re
import spacy
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordNERTagger


java_path = r"C:/Program Files/Java/jdk-21/bin/java.exe"
os.environ['JAVAHOME'] = java_path

stanford_dir = os.path.abspath(os.path.join(os.getcwd(), 'stanford-ner-2020-11-17'))
jarfile = os.path.join(stanford_dir, 'stanford-ner.jar')
modelfile = os.path.join(stanford_dir, 'classifiers', 'english.all.3class.distsim.crf.ser.gz')


def preprocess_text(raw_text:str):
    """
    Removing unnecessary characters and white space
    Also removing 's as this seems to confuse the NER model
    """
    response_pretty = BeautifulSoup(raw_text, "html.parser")
    page_text = response_pretty.get_text()
    page_text = re.sub(r"[^\x00-\x7F]+", "", page_text)
    page_text = page_text.replace("\r\n", " ").replace("'s", " ")
    page_text = re.sub(r"\s+", " ", page_text)


def remove_stopwords(raw_text:str):
    words = word_tokenize(raw_text)
    stop_words = set(stopwords.words('english'))
    filtered_text = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_text)

    return filtered_text

def get_persons_and_positions(results, text):
    person_and_positions = []

    start_position = 0
    end_position = 0

    for word, tag in results:
        if tag == 'PERSON':
            start_position = text.find(word, end_position)
            end_position = start_position + len(word)
            
            person_and_positions.append({'name': word, 'position':(start_position,end_position)})


    return person_and_positions

def get_person_counts(fullnames_and_positions):
    person_counts = {}
    for person in fullnames_and_positions:
        if person['name'] not in person_counts.keys():
            person_counts[person['name']] = {}
            person_counts[person['name']]['count'] = 1
            person_counts[person['name']]['position'] = [person['position']]
        else:
            person_counts[person['name']]['count'] += 1
            person_counts[person['name']]['position'].append(person['position'])

    return sorted(person_counts.items(), key=lambda x: x[1]['count'], reverse=True)

def get_fullnames(persons_and_positions):
    fullnames_and_positions = []
    i = 0
    while i < len(persons_and_positions):
        current = persons_and_positions[i]
        name = current['name']
        start, end = current['position']
        
        while i < len(persons_and_positions) - 1 and persons_and_positions[i + 1]['position'][0] - end == 1:
            i += 1
            next_item = persons_and_positions[i]
            name += " " + next_item['name']
            end = next_item['position'][1]
        
        fullnames_and_positions.append({'name': name, 'position': (start, end)})
    
        i += 1

    return fullnames_and_positions
    


def get_section_of_text(preprocessed_text,start, end):
    before_name = preprocessed_text[:start].split()[-100:]
    after_name = preprocessed_text[end:].split()[:101]

    return " ".join(before_name) + " ".join(after_name)

def get_associated_places_counts(person_counts:list, preprocessed_text:str, nlp):
    """
    Getting associated palaces for each instance of person and the counts
    """
    for entry in person_counts:
        entry[1]['associated_places'] = {}
        for pos in entry[1]['position']:
            start, end = pos
            section_of_text = get_section_of_text(preprocessed_text, start, end)
            doc = nlp(section_of_text)

            for ent in doc.ents:
                if ent.label_ == 'GPE':
                    if ent.text not in entry[1]['associated_places'].keys():
                        entry[1]['associated_places'][ent.text] = 1
                    else:
                        entry[1]['associated_places'][ent.text] += 1
    
    return person_counts

def order_associated_places(full_counts:list):
    """
    Sorting list of places by the number of times it appears with each person
    """
    for person in full_counts:
        person["associated_places"] = sorted(person["associated_places"], key=lambda x: x["count"], reverse=True)

    return full_counts

def format_list(full_counts:list):
    """
    Formatting list into the required form as stated in test instructions
    """
    people = []

    for i, entry in enumerate(full_counts):
        people.append({})
        people[i]["name"] = entry[0]
        people[i]["count"] = entry[1]['count']
        people[i]["associated_places"] = []

        for entry_ in entry[1]['associated_places'].items():
            place = {}
            k, v = entry_
            place['name'] = k
            place['count'] = v
            people[i]["associated_places"].append(place)
    people = order_associated_places(people)
    return people





def get_response(url:str):
    # get text from page
    response = requests.get(url)
    if response.status_code == 200:
        raw_text = response.text
    else:
        return ['Invalid URL']

    # load in tagger 
    st = StanfordNERTagger(modelfile, jarfile)

    # preprocess text
    filtered_text = remove_stopwords(raw_text)

    tokenized_text_ = nltk.word_tokenize(filtered_text)
    results = st.tag(tokenized_text_)

    # use Stanford NER
    # get persons and their counts
    person_and_positions = get_persons_and_positions(results, filtered_text)
    fullnames_and_positions = get_fullnames(person_and_positions)
    person_counts = get_person_counts(fullnames_and_positions)

    # load spacy model for NER on locations
    nlp = spacy.load("en_core_web_md")

    # get places associated with each person and their counts
    persons_and_associated_places = get_associated_places_counts(person_counts, filtered_text, nlp)

    # format list according to requirements
    people = format_list(persons_and_associated_places)

    return people

