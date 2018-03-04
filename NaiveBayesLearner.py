import nltk
import math
import os
from nltk.tokenize import RegexpTokenizer
import filecmp
import io
import json

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
            tokens.append(word)

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
        for word in vocabulary:
            if(word not in conditional_probability):
                conditional_probability[word] = {}
            conditional_probability[word][cls] = (float(word_count[word] + 1))/denominator

    return (vocabulary, prior, conditional_probability)

def test(classes, vocabulary, prior, conditional_probability, document_text):

    relevant_words = extract_tokens_from_document(vocabulary, document_text)
    scores = {}

    for cls in classes:
        scores[cls] = math.log(prior[cls])
        for word in relevant_words:
            if(cls in conditional_probability[word]):
                scores[cls] += math.log(conditional_probability[word][cls])
    
    max_key = max(scores, key=scores.get)

    return (max_key, scores[max_key])


def get_documents_dictionary(file_names, folder, cls):
    for file_name in file_names:
        with io.open(folder + file_name, encoding = 'utf-8') as current_file:
            data=current_file.read().replace('\n', '')
            documents[data] = cls
    return documents

def file_cmp(file_names):
    for i in range(0, len(file_names)):
        for j in range(i + 1, len(file_names)):
            if(filecmp.cmp(file_names[i], file_names[j])):
                print("File Contents duplicated")

def predict_class_for_test_data(file_names, test_folder, class_to_test_for):
    correct_predictions = 0
    incorrect_predictions = 0
    for file_name in file_names:
        with io.open(test_folder + file_name, encoding = 'ISO-8859-1') as current_file:
            current_string = current_file.read()
            current_text = extract_vocabulary([current_string])
            test_result = test(classes, vocabulary, prior, conditional_probability, current_text)
            if(test_result[0] == class_to_test_for):
                correct_predictions = correct_predictions + 1
            else:
                incorrect_predictions = incorrect_predictions + 1
    return  correct_predictions, incorrect_predictions

if __name__ == "__main__":
    classes = ['ham', 'spam']
    
    vocabulary = set()
    prior = {}
    conditional_probability = {}

    dataset1_training_ham_folder = "hw2_train/train/ham/"
    dataset1_training_spam_folder = "hw2_train/train/spam/"
    dataset1_test_ham_folder = "hw2_test/test/ham/"
    dataset1_test_spam_folder = "hw2_test/test/spam/"

    dataset2_training_ham_folder = "enron1_train/enron1/train/ham/"
    dataset2_training_spam_folder = "enron1_train/enron1/train/spam/"
    dataset2_test_ham_folder = "enron1_test/enron1/test/ham/"
    dataset2_test_spam_folder = "enron1_test/enron1/test/spam/"

    dataset2_training_ham_folder = "enron1_train/enron1/train/ham/"
    dataset2_training_spam_folder = "enron1_train/enron1/train/spam/"
    dataset2_test_ham_folder = "enron1_test/enron1/test/ham/"
    dataset2_test_spam_folder = "enron1_test/enron1/test/spam/"

    dataset3_training_ham_folder = "enron4_train/enron4/train/ham/"
    dataset3_training_spam_folder = "enron4_train/enron4/train/spam/"
    dataset3_test_ham_folder = "enron4_test/enron4/test/ham/"
    dataset3_test_spam_folder = "enron4_test/enron4/test/spam/"

    dataset1_trained_file = "ds1_trained_dictionary.json";
    dataset2_trained_file = "ds2_trained_dictionary.json";
    dataset3_trained_file = "ds3_trained_dictionary.json";

    dataset1_prior_file = "ds1_trained_prior_values.json"
    dataset2_prior_file = "ds2_trained_prior_values.json"
    dataset3_prior_file = "ds3_trained_prior_values.json"

    dataset_to_use = int(input("Which dataset do you want to use from the H.W web page? Enter 1 or 2 or 3"))
    
    ham_folder = ""
    spam_folder = ""
    ham_test_folder = ""
    spam_test_folder = ""

    if(dataset_to_use == 1):
        ham_folder = dataset1_training_ham_folder
        spam_folder = dataset1_training_spam_folder
        ham_test_folder = dataset1_test_ham_folder
        spam_test_folder = dataset1_test_spam_folder
        training_file = dataset1_trained_file
        prior_file = dataset1_prior_file

    if(dataset_to_use == 2):
        ham_folder = dataset2_training_ham_folder
        spam_folder = dataset2_training_spam_folder
        ham_test_folder = dataset2_test_ham_folder
        spam_test_folder = dataset2_test_spam_folder
        training_file = dataset2_trained_file
        prior_file = dataset2_prior_file

    if(dataset_to_use == 3):
        ham_folder = dataset3_training_ham_folder
        spam_folder = dataset3_training_spam_folder
        ham_test_folder = dataset3_test_ham_folder
        spam_test_folder = dataset3_test_spam_folder
        training_file = dataset3_trained_file
        prior_file = dataset3_prior_file

    if(input("Do you want to retrain from scratch?") == "yes"):
        documents = {}    

        ham_file_names = os.listdir(ham_folder)
        cls = 'ham'
        #file_cmp([ham_folder + file_name for file_name in ham_file_names])
        documents.update(get_documents_dictionary(ham_file_names, ham_folder, cls))

        
        spam_file_names = os.listdir(spam_folder)
        cls = 'spam'
        #file_cmp([spam_folder + file_name for file_name in spam_file_names])
        documents.update(get_documents_dictionary(spam_file_names, spam_folder, cls))

        vocabulary, prior, conditional_probability = train(classes, documents)
        
        conditional_probability_file = open("trained_dictionary.json", "w")
        conditional_probability_file.write(json.dumps(conditional_probability))

    else:
        vocabulary_file = open("vocabulary.txt", "r") 
        vocabulary = set([word.replace("'","") for word in vocabulary_file.read().split(',')])
        vocabulary = set(list(word.strip() for word in vocabulary))
        vocabulary_file.close()

        prior_file = open("trained_prior_values.json", "r")
        prior = json.loads(prior_file.read())
        prior_file.close()

        conditional_probability_file = open("trained_dictionary.json", "r")
        conditional_probability = json.loads(conditional_probability_file.read())
        conditional_probability_file.close()
    
    
    test_ham_file_names = os.listdir(ham_test_folder)

    ham_correct_predictions, ham_incorrect_predictions = predict_class_for_test_data(test_ham_file_names, ham_test_folder, 'ham')
    
    spam_test_folder = 'hw2_test/test/spam/'
    test_spam_file_names = os.listdir(spam_test_folder)

    spam_correct_predictions, spam_incorrect_predictions = predict_class_for_test_data(test_spam_file_names, spam_test_folder, 'spam')

    accuracy = float(ham_correct_predictions + spam_correct_predictions) / (ham_correct_predictions + ham_incorrect_predictions + spam_correct_predictions + spam_incorrect_predictions)

    print(accuracy)

    