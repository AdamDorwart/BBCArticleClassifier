from sklearn.naive_bayes import MultinomialNB
from bbc import load_bbc
import numpy as np
import random

vocab, X, y, keys = load_bbc()
class sports:
   vocab
   X
   y
   keys

sports.vocab, sports.X, sports.y, sports.keys = load_bbc(dataset="sports")

classif = MultinomialNB()
classif.fit(X, y)

sports.classif = MultinomialNB()
sports.classif.fit(sports.X, sports.y)

featureVector = np.zeros(len(vocab))
newsArticle = open("article.txt", 'r')
for line in newsArticle:
   for word in line.replace('\n','').split(' '):
      if word in vocab:
         featureVector[vocab[word]] += 1
newsArticle.close()

test = random.randint(0,len(X))
print "Random article class: " + keys[y[test]]
print "Random article prediction: " + keys[classif.predict(X[test])]

test = random.randint(0,len(sports.X))
print "Random sports article class: " + sports.keys[sports.y[test]]
print "Random sports article prediction: " + sports.keys[sports.classif.predict(sports.X[test])]

print "Given News Article Prediction: " + keys[classif.predict(featureVector)]
if keys[classif.predict(featureVector)] == "sport":
   sports.featureVector = np.zeros(len(sports.vocab))
   newsArticle = open("article.txt", 'r')
   for line in newsArticle:
      for word in line.replace('\n','').split(' '):
         if word in sports.vocab:
            sports.featureVector[sports.vocab[word]] += 1
   newsArticle.close()
   print "Sports subclass Prediction: " + sports.keys[sports.classif.predict(sports.featureVector)]

wrong = 0;
for i in range(0,len(y)-1):
   if classif.predict(X[i]) != y[i]:
      wrong = wrong + 1
print "Test error: " + str((float(wrong)/(len(y)-1)*100)) + "%"

wrong = 0;
for i in range(0,len(sports.y)-1):
   if sports.classif.predict(sports.X[i]) != sports.y[i]:
      wrong = wrong + 1
print "Sports test error: " + str((float(wrong)/(len(sports.y)-1)*100)) + "%"