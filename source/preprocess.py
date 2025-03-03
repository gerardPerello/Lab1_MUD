from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import jieba  # Tokenization for Chinese
from janome.tokenizer import Tokenizer  # Tokenization for Japanese

# Tokenizer function. You can add here different preprocesses.
def detect_language(text):
    '''
    Task: Detect the language of the sentence based on Unicode characters
    and a fallback to `langdetect` for more accurate language detection.

    Input: Sentence in string format
    Output: Language type (chinese, japanese, korean, swedish, other)
    '''
    chinese_range = r'[\u4e00-\u9fff]'  # Chinese characters range
    japanese_range = r'[\u3040-\u30ff\u31f0-\u31ff]'  # Hiragana, Katakana and other Japanese characters
    korean_range = r'[\uac00-\ud7af]'  # Hangul characters for Korean

    # Check for Chinese characters using Unicode range
    if re.search(chinese_range, text):
        return "chinese"
    # Check for Japanese characters using Unicode range
    elif re.search(japanese_range, text):
        return "japanese"
    # Check for Korean characters using Unicode range
    elif re.search(korean_range, text):
        return "korean"
    else:
        return "other"

# Main preprocessing function
def preprocess(sentence, labels=None, lowercase=True, remove_punctuation=True, 
               remove_numbers=True, tokenize=True, lemmatize=True, use_unified_tokenizer=True):
    '''
    Task: Given a sentence, apply all the required preprocessing steps
    to compute features for training a classifier. This includes sentence 
    splitting, tokenization, and optional steps like lemmatization.

    Input: 
    - Sentence (string format)
    - Optional flags for preprocessing steps like lowercase, punctuation removal, etc.
    
    Output: 
    - Preprocessed sentence in string format
    '''
    return sentence
    # Optional step: Convert text to lowercase
    if lowercase:
        sentence = sentence.lower()
    
    # Optional step: Remove punctuation
    if remove_punctuation:
        sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
    
    # Optional step: Remove numbers
    if remove_numbers:
        sentence = re.sub(r'\d+', '', sentence)  # Remove numbers
    
    # Detect the language of the sentence
    lang = detect_language(sentence)
    
    # Tokenization process based on the language or using a unified tokenizer
    if tokenize:
        if use_unified_tokenizer:
            # Apply standard word-level tokenization for all languages
            sentence = " ".join(word_tokenize(sentence))
        else:
            # Apply specific tokenization for different languages
            if lang == "chinese":
                sentence = " ".join(jieba.cut(sentence))  # Tokenization for Chinese
            elif lang == "japanese":
                tokenizer = Tokenizer()
                sentence = " ".join(token.surface for token in tokenizer.tokenize(sentence))  # Tokenization for Japanese
            elif lang == "korean":
                from konlpy.tag import Okt
                okt = Okt()
                sentence = " ".join(okt.morphs(sentence))  # Tokenization for Korean
            else:
                sentence = " ".join(word_tokenize(sentence))  # Default tokenization for other languages

    # Lemmatization step only for morphological languages like English
    if lemmatize and lang == "other":
        lemmatizer = WordNetLemmatizer()
        sentence = " ".join([lemmatizer.lemmatize(word) for word in sentence.split()])

    return sentence

