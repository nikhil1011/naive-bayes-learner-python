import nltk
import math
import os
from nltk.tokenize import RegexpTokenizer

def get_tokens_in_document(document):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(document)
    return tokens

def extract_vocabulary(documents):
    vocabulary = set()
    for document in documents:
        tokens = get_tokens_in_document(document)
        #tokens = nltk.word_tokenize(document)
        for token in tokens:
            vocabulary.add(token)
    return vocabulary

def count_documents_of_a_class(cls, documents):
    count = 0
    for document in documents:
        if(documents[document] == cls):
            count = count + 1
    return count

def concatenate_text_of_class(documents, cls):
    text = []

    for document in documents:
        if(documents[document] == cls):
            text = text + get_tokens_in_document(document)

    return text

def count_word_occurences_in_text(word, text):
    count = 0

    for text_word in text:
        if(word == text_word):
            count = count + 1

    return count

def calculate_denominator(word_count):
    total = 0
    for word in word_count:
        total = total + word_count[word] + 1
    return total

def extract_tokens_from_document(vocabulary, document_text):
    tokens = []

    for word in document_text:
        if(word in vocabulary):
            tokens.add(word)

    return tokens

def train(classes, documents):

    vocabulary = extract_vocabulary(documents)
    n = len(documents)

    prior = {}
    concatenated_text = []
    word_count = {}
    conditional_probability = {}

    for cls in classes:
        prior[cls] = float(count_documents_of_a_class(cls, documents))/n
        text_of_class = concatenate_text_of_class(documents, cls)
        for word in vocabulary:
            word_count[word] = count_word_occurences_in_text(word, text_of_class)
        denominator = calculate_denominator(word_count)
        conditional_probability[word] = {}
        conditional_probability[cls] = (float(word_count[word] + 1))/denominator

    return (V, prior, conditional_probability)

def test(classes, vocabulary, prior, conditional_probability, document_text):

    relevant_words = extract_tokens_from_document(vocabulary, document_text)
    scores = {}

    for cls in classes:
        score[cls] = math.log(prior[cls])
        for word in relevant_words:
            score[cls] += math.log(conditional_probability[word][cls])
    
    max_key, max_value = max(stats.iteritems(), key=lambda x:x[1])

    return (max_key, max_value)


if __name__ == "__main__":
    ham_folder = 'hw2_train/train/ham'
    ham_file_names = os.listdir(ham_folder)
    classes = ['ham', 'spam']
    