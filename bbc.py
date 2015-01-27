import os, struct
import numpy as np
from numpy import zeros

def load_bbc(dataset="", path="."):
   """
   Loads BBC classified article dataset

   http://mlg.ucd.ie/datasets/bbc.html
   """

   if dataset == "":
      artPath = os.path.join(path, 'bbc.mtx')
      vocPath = os.path.join(path, 'bbc.terms')
      labPath = os.path.join(path, 'bbc.classes')
   elif dataset == "sports":
      artPath = os.path.join(path, 'bbcsport.mtx')
      vocPath = os.path.join(path, 'bbcsport.terms')
      labPath = os.path.join(path, 'bbcsport.classes')
   else:
      raise ValueError("dataset must be '' or 'sports'")

   vocab = {}
   counter = 0
   fvocab = open(vocPath, 'r')
   for line in fvocab:
      vocab[line.replace('\n','')] = counter
      counter += 1
   fvocab.close()

   farticles = open(artPath, 'r')
   # Eat up first line
   farticles.readline()
   # Read sizing data
   sizing = farticles.readline().split(" ");
   articles = np.zeros((int(sizing[1]), int(sizing[0])), dtype=np.uint16)
   for line in farticles:
      point = line.replace('\n','').split(" ")
      articles[int(point[1])-1][int(point[0])-1] = float(point[2])
   farticles.close();
   flabels = open(labPath, 'r')
   # Eat up first 2 lines
   flabels.readline(); flabels.readline();
   # Read key translations 
   keyValues = flabels.readline().replace('\n','');
   keys = keyValues.split(" ")[2].split(",")
   #Eat up 1 more line
   flabels.readline()
   labels = np.zeros((int(sizing[1])), dtype=np.uint8)
   for line in flabels:
      point = line.replace('\n','').split(" ")
      labels[int(point[0])] = int(point[1])
   flabels.close()

   return vocab, articles, labels, keys