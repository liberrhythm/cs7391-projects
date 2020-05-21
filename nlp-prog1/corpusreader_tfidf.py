from nltk.corpus import stopwords
import numpy as np
from nltk import FreqDist
from nltk.stem import PorterStemmer
import re

class CorpusReader_TFIDF:

    # constructor
    def __init__(self, corpus, tf='raw', idf='base', stopword='none', stemmer=PorterStemmer(), ignorecase='yes'):
        # set class parameter variables
        self.corpus = corpus
        self.tf = tf
        self.idf = idf
        self.stopword = stopword
        self.stemmer = stemmer
        self.ignorecase = ignorecase

        # set regex for preprocessing
        if self.ignorecase == 'yes':
            self.regex = re.compile('[^a-z]')
        else:
            self.regex = re.compile('[^a-zA-Z]')

        # use helper functions to initialize helpful variables and tf-idf vector matrix
        self.swlist = self.__get_stopwords()
        self.docs = np.array(corpus.fileids())
        self.num_docs = len(self.docs)

        # get unique words (dimension vector) and dictionary with word as key and index as value
        self.unique_words = self.__get_unique_words()
        self.word_dict = self.__get_word_dict()

        # calculate and get tf / idf / tf-idf vector and matrix values
        self.tf_vectors = self.__get_tf()
        self.idf_vec = self.__get_idf()
        self.tfidf_vectors = self.__get_tfidf()


    # standard corpus reader methods

    def corpus(self):
        return self.corpus

    def fileids(self):
        return self.corpus.fileids()

    def raw(self, fileids=None):
        if fileids is not None:
            return self.corpus.raw(fileids=fileids)
        else:
            return self.corpus.raw()

    def words(self, fileids=None):
        if fileids is not None:
            return self.corpus.words(fileids=fileids)
        else:
            return self.corpus.words()

    def open(self, fileid):
        return self.corpus.open(fileid=fileid)

    def abspath(self, fileid):
        return self.corpus.abspath(fileid=fileid)


    # private helper methods

    # returns a preprocessed list of words, taking care of case, punctuation, stemming, and / or stopwords
    def __preprocess(self, words):
        # handle punctuation
        words = [self.regex.sub('', word) for word in words]

        # stem words
        words = [self.stemmer.stem(word) for word in words if word]

        # check for stopwords and remove any empty strings
        words = [word for word in words if word and len(word) > 2 and not word in self.swlist]

        return words

    # returns set of stopwords (empty, nltk default, or through text file)
    def __get_stopwords(self):
        if (self.stopword == 'none'):
            # return empty set if no stopwords provided
            return set()
        elif (self.stopword == ''):
            # return default nltk set of english stopwords
            return set(stopwords.words('english'))
        else:
            # populate stopword set from text file
            with open(self.stopword) as f:
                stopword_list = [self.regex.sub('', word) for line in f
                                 for word in re.split('[;,.\-\n ]', line) if word]
                print(stopword_list)

            f.close()
            return set(stopword_list)

    # returns a sorted list of unique words in the given corpus
    def __get_unique_words(self):
        all_words = self.__preprocess(set(self.corpus.words()))
        return sorted(set(all_words))

    # returns a dictionary with word keys and list index values for quicker access in tf-idf method
    def __get_word_dict(self):
        return {k: v for v, k in enumerate(self.unique_words)}

    # calculates and returns term frequency vectors for all documents and unique words in the corpus
    def __get_tf(self):
        # declare term frequency matrix of zeroes
        tf_vectors = np.zeros(shape=(len(self.docs), len(self.unique_words)))

        doc_idx = 0
        for doc in self.docs:
            # get frequency distribution of words for a given document
            doc_words = self.corpus.words(fileids=[doc])
            doc_words = self.__preprocess(doc_words)
            doc_freq = FreqDist(doc_words)

            # set frequencies in term frequency matrix for each word in the document
            for word in doc_freq:
                word_idx = self.word_dict[word]
                tf_vectors[doc_idx][word_idx] = doc_freq[word]

            doc_idx += 1

        # alter and return term frequency matrix based on given tf parameter
        if (self.tf == 'log'):
            tf_vectors_log = np.log2(tf_vectors, where=(tf_vectors > 0))
            tf_vectors_log[tf_vectors > 0] += 1
            return tf_vectors_log
        elif (self.tf == 'binary'):
            tf_vectors_binary = (tf_vectors > 0).astype(int)
            return tf_vectors_binary
        else:
            return tf_vectors

    # calculates and returns term frequency vectors for all unique words in the corpus
    def __get_idf(self):
        # declare inverse document frequency vector of zeroes
        idf_vec = np.zeros(shape=len(self.unique_words))

        # calculate base inverse document frequency values for each word
        counter = 0
        for word_vec in self.tf_vectors.T:
            idf_vec[counter] = np.count_nonzero(word_vec)
            counter += 1

        # alter and return inverse document frequency vector based on given idf parameter
        if (self.idf == 'smooth'):
            idf_vec = self.num_docs / idf_vec
            idf_vec_smooth = np.log2(idf_vec + 1)
            return idf_vec_smooth
        else:
            idf_vec = self.num_docs / idf_vec
            idf_vec = np.log2(idf_vec)
            return idf_vec

    # calculate and return final tf-idf matrix
    def __get_tfidf(self):
        return self.tf_vectors * self.idf_vec


    # tf-idf specific public methods

    # returns a list of all tf-idf vectors corresponding to all files, one file, or for a list of files in the corpus
    # vector results are ordered by the order in which fileids are returned
    # optional parameters: fileid and filelist
    def tf_idf(self, fileid=None, filelist=None):
        if fileid is not None:
            # return tf-idf vector for a specific file
            indices = np.where(self.docs == fileid)[0]
            doc_idx = indices[0]
            return self.tfidf_vectors[doc_idx, :]
        elif filelist is not None:
            # return tf-idf vectors for a list of files
            filemask = np.isin(self.docs, filelist)
            indices = np.where(filemask)[0]
            return self.tfidf_vectors(indices)
        else:
            # return tf-idf vectors for all files
            return self.tfidf_vectors

    # returns the list of the words in the order of the dimension of each tf-idf vector
    def tf_idf_dim(self):
        return self.unique_words

    # returns a vector corresponding to the tf_idf vector for a new document (provided as a list of words)
    # new document uses the same stopword, stemming, ignorecase treatment as the original corpus
    # idf for the original corpus is used to calculate the results
    def tf_idf_new(self, words):
        # declare tf vector for new document and get frequency distribution of document words
        tf_vec = np.zeros(shape=len(self.unique_words))
        doc_freq = FreqDist(self.__preprocess(words))

        # assign frequencies in tf vector
        for word in doc_freq:
            if word in self.word_dict:
                word_idx = self.word_dict[word]
                tf_vec[word_idx] = doc_freq[word]

        # calculate and return tf-idf vector for new document using original idf vector
        return tf_vec * self.idf_vec

    # returns the cosine similarity between two documents in the corpus
    def cosine_sim(self, fileid):
        # use tf_idf method to get tf-idf vectors for specified files
        vec1 = self.tf_idf(fileid=fileid[0])
        vec2 = self.tf_idf(fileid=fileid[1])

        # calculate cosine similarity between the two documents
        numerator = np.dot(vec1, vec2)
        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return numerator / denominator

    # returns the cosine similarity between a document in the corpus and a new document (provided as a list of words)
    # new document uses the same stopword, stemming, ignorecase treatment as the original corpus
    # idf for the original corpus is used to calculate the results
    def cosine_sim_new(self, words, fileid):
        # get tf-idf vectors for new and existing files
        vec1 = self.tf_idf_new(words)
        vec2 = self.tf_idf(fileid=fileid)

        # calculate cosine similarity between the two documents
        numerator = np.dot(vec1, vec2)
        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return numerator / denominator