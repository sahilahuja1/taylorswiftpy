import numpy as np
import numpy as np
import matplotlib.pyplot as plt

import madmom
from madmom.audio import chroma
from madmom.features import chords
from madmom.audio import ffmpeg

from sklearn import svm
from sklearn.neural_network import MLPClassifier

text_file = open('covers/list1.list', 'r')
actualNames = text_file.readlines()
text_file = open('covers/list2.list', 'r')
coverNames = text_file.readlines()

actuals = []
covers = []

numsongs = len(actualNames)

for i in range(numsongs):
	actualNames[i] = 'covers/' + actualNames[i][:-1] + '.txt'
	coverNames[i] = 'covers/' + coverNames[i][:-1] + '.txt'

	actual = np.loadtxt(actualNames[i], dtype = np.float).astype(float)
	cover = np.loadtxt(coverNames[i], dtype = np.float).astype(float)

	actuals.append(actual[500:1000].flatten().tolist())
	covers.append(cover[500:1000].flatten().tolist())

# allSongs = actuals + covers
# print(len(allSongs))

# kmeans = KMeans(n_clusters = 80).fit(allSongs)
# print(kmeans.labels_[0:80])
# print(kmeans.labels_[80:160])

X = []
y = []

numberRandom = 1

for i in range(60):

	X.append(np.subtract(actuals[i], covers[i]).tolist())
	y.append(1)

	myrange = list(range(0,i)) + list(range(i+1,numsongs))
	randomSelection = np.random.choice(myrange, numberRandom)

	for j in randomSelection:
		X.append(np.subtract(actuals[i], covers[j]).tolist())
		y.append(0)

clf = MLPClassifier(hidden_layer_sizes=(12, 12), random_state=1)
clf.fit(X, y)  
print("Training Score: ", clf.score(X,y))

Xtest = []
ytest = []
testingActuals = []
testingCovers = []

for i in range(60, numsongs):

	Xtest.append(np.subtract(actuals[i], covers[i]).tolist())
	ytest.append(1)
	testingActuals.append(i)
	testingCovers.append(i)

	myrange = list(range(0,i)) + list(range(i+1,numsongs))
	randomSelection = np.random.choice(myrange, numberRandom)

	print(randomSelection)

	for j in randomSelection:
		Xtest.append(np.subtract(actuals[i], covers[j]).tolist())
		ytest.append(0)
		testingActuals.append(i)
		testingCovers.append(j)

yprediction = clf.predict(Xtest)
print("Testing Score: ", clf.score(Xtest, ytest))

for i in range(len(Xtest)):
	if (yprediction[i] == ytest[i] and ytest[i] == 1):
		print(ytest[i])
		print(i)
		print(actualNames[testingActuals[i]][:-4])
		print(coverNames[testingCovers[i]][:-4])
		print("---------")
