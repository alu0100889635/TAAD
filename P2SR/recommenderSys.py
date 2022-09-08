import sys
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


with open(sys.argv[1], 'r') as f:
    liked = [line.rstrip() for line in f]

documents = list(map(lambda x: re.split(r"\:\s|\/", x, 2), liked))
dfdocs = pd.DataFrame(documents, columns=['DocNumb', 'Document', 'Like'])

def vectorizeDocs(doc):
    vectorizer = TfidfVectorizer(stop_words = "english")
    return vectorizer.fit_transform(doc), vectorizer.get_feature_names_out()

tfidfmatrix, words = vectorizeDocs(dfdocs['Document'])

tfidfdoc = []
for i in tfidfmatrix.toarray():
    tfidfdocnonzero = []
    for x in range(0, len(i)):
        if(i[x]) != 0.0:
            tfidfdocnonzero.append(i[x])
    tfidfdoc.append(tfidfdocnonzero)

test = []
for i in documents:
    test.append([i[1]])

words = []
positions = []
for i in test:
    X, word = vectorizeDocs(i)
    position = []
    for w in word:
        position.append(i[0].lower().find(w))
    positions.append(position)
    words.append(word)

dfdocs['TF-IDF'] = tfidfdoc
dfdocs['Term Ind'] = positions
dfdocs['Terms'] = words
print(dfdocs[['DocNumb', 'Term Ind', 'Terms', 'TF-IDF', 'Like']])

cosine_similarities = cosine_similarity(tfidfmatrix, tfidfmatrix)
lowerTriangleMatrix = np.tril(cosine_similarities)

dfmatrix = pd.DataFrame(lowerTriangleMatrix)

sorted_values = pd.Series

for i in range(0, len(dfmatrix.columns)):
    if dfdocs.iloc[i]['Like'] == "1":
        print("\nLe ha gustado el documento ", dfdocs.iloc[i]['DocNumb'],": ")
        sorted_values = dfmatrix[i].sort_values(ascending = False)
        for index, value in sorted_values.items():
            if(round(value, 6) != 1.0): 
                if(round(value, 6) >= 0.0): 
                    if(dfdocs.iloc[index]['Like'] == "N"): 
                        print("Documento ", dfdocs.iloc[index]['DocNumb'],  
                        " -> Similitud con documento", dfdocs.iloc[i]['DocNumb'], "= ", round(value, 6))
