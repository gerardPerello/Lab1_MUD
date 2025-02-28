import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')


def preprocess(sentence, labels, lowercase=False, remove_punctuation=False, remove_numbers=False, tokenize=True, lemmatize=False):
    

    if lowercase:
        sentence = sentence.lower()
    
    if remove_punctuation:
        sentence = re.sub(r'[^\w\s]', '', sentence)
    
    if remove_numbers:
        sentence = re.sub(r'\d+', '', sentence)
    sentence = [sentence]
    if tokenize:
        sentence = [word_tokenize(sent) for sent in sentence]
    
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        sentence = [[lemmatizer.lemmatize(word) for word in sent] for sent in sentence]
    
    sentence = ' '.join(sentence[0])
    
    return sentence


