from sklearn.naive_bayes import MultinomialNB
from bbc import load_bbc
import numpy as np
import random

vocab, X, y, keys = load_bbc()

classif = MultinomialNB()
classif.fit(X, y)

featureVector = np.zeros(len(X[0]))
newsArticle = open("article.txt", 'r')
for line in newsArticle:
   for word in line.replace('\n','').split(' '):
      if word in vocab:
         featureVector[vocab[word]] += 1

test = random.randint(0,len(X))
print "Random article class: " + keys[y[test]]
print "Random article prediction: " + keys[classif.predict(X[test])]

print "Given News Article Prediction: " + keys[classif.predict(featureVector)]

wrong = 0;
for i in range(0,len(y)-1):
   if classif.predict(X[i]) != y[i]:
      wrong = wrong + 1
print "Test error: " + str((float(wrong)/(len(y)-1)*100)) + "%"