import nltk
from nltk.corpus import stopwords
import sys
import os
import string
import math
from collections import Counter

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)
    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    results = dict()
    for filenames in os.listdir(directory):
        if filenames.endswith(".txt"):
            with open(os.path.join(directory, filenames)) as f:
                contents = f.read()
                results[filenames] = contents
    return results


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    filtered_txt = []
    stop_wrds = set(stopwords.words('english'))
    for word in nltk.word_tokenize(document):
        if word not in stop_wrds and word not in string.punctuation:
            filtered_txt.append(word.lower())
    return filtered_txt

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    word_idf = dict()
    doc_count = 0
    words = set()
    num_docs = len(documents)
    # for each word that is in a document, count the number of documents that word appears
    for key, value in documents.items():
        words.update(value)
    for word in words:
        # iterate over documents and check if word is in that document
        # if it is in document increment doc_count by 1
        word_count = sum([1 for doc in documents if word in documents[doc]])
        # count idf
        idf = math.log(num_docs/word_count)
        word_idf[word] = idf
    return word_idf


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # return a list if sorted filenames by their tf-idf score.
    def tf_idf_score(file):
        return sum(files[file].count(word) * idfs[word] for word in query)
    tp_files = sorted(files, key = tf_idf_score, reverse = True)
    return tp_files[:n]

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # return a list of sorted sentences by their idf score.
    def idf_score(sentence):
        return sum(idfs[word] for word in query if word in sentences[sentence])
    # if two sentences have matching term measure break the tie by matching by query term density
    def query_term_density(sentence):
        return sum([1 for word in sentences[sentence] if word in query])/len(sentence)
    # created tuple_key function to return both idf_score and query_term_density due to the fact that
    # key = lambda i: (idf_score, query_term_density) in tp_sentences did not work for unknown reason
    def tuple_key(sentence):
        return (idf_score(sentence), query_term_density(sentence))
    tp_sentences = sorted(sentences.keys(), key = tuple_key, reverse = True)
    return tp_sentences[:n]


if __name__ == "__main__":
    main()
