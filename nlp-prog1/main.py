from nltk.corpus import brown, state_union
from corpusreader_tfidf import CorpusReader_TFIDF
import itertools
import time

def main():
    # Brown
    start = time.perf_counter()

    corpus1 = brown
    cr1 = CorpusReader_TFIDF(corpus1)

    print("Brown")
    print(cr1.tf_idf_dim()[0:15])

    corp_docs = cr1.fileids()[0:5]
    for doc in corp_docs:
        print(doc + ', ' + str(cr1.tf_idf(fileid=doc)[0:15]))

    for doc in corp_docs:
        print(doc + ' ' + doc + ': ' + str(cr1.cosine_sim([doc, doc])))

    doc_pairs = list(itertools.combinations(corp_docs, 2))
    for pair in doc_pairs:
        print(pair[0] + ' ' + pair[1] + ': ' + str(cr1.cosine_sim(pair)))

    end = time.perf_counter()
    print(end - start)

    print("\n\n")

    # State of the Union
    start = time.perf_counter()

    corpus2 = state_union
    cr2 = CorpusReader_TFIDF(corpus2)

    print("State of the Union")
    print(cr2.tf_idf_dim()[0:15])

    corp_docs = cr2.fileids()[0:5]
    for doc in corp_docs:
        print(doc + ', ' + str(cr2.tf_idf(fileid=doc)[0:15]))

    for doc in corp_docs:
        print(doc + ' ' + doc + ': ' + str(cr2.cosine_sim([doc, doc])))

    doc_pairs = list(itertools.combinations(corp_docs, 2))
    for pair in doc_pairs:
        print(pair[0] + ' ' + pair[1] + ': ' + str(cr2.cosine_sim(pair)))

    end = time.perf_counter()
    print(end - start)

main()