# Word Vector Space Abstraction

## Installation

Install python-2.7

sudo pip install numpy pandas numba
sudo apt install libopenblas-dev


### Usage of wordvecspace library

#### Export the path of data files to the environment variables
$export data_dir="/home/ram/alpha/data/w2v_new_Google/shard_0"
$python

```python=!

#Import classes from wordvecspace library
>>>from wordvecspace import WordVecSpace, UnknownWord, UnknownIndex

#Read the path of vectors.npy and VOCAB.txt from environment variables
>>>data_dir = os.environ['data_dir']

#Create Instace for WordVecSpace
>>>wv = WordVecSpace(data_dir)


#Load the data files into memory
>>>wv.load()

#Check word exist or not in wordvecspce
>>>print wv.does_word_exist("india")
True

>>>print wv.does_word_exist("India")
False


#Get Index of the word
>>> try:
...     print wv.get_word_index("iskjf4s")
... except UnknownWord, e:
...     print "Word %s was not found" % e.word
... 
Word iskjf4s was not found
>>> try:
...     print wv.get_word_index("for")
... except UnknownWord, e:
...     print "Word %s was not found" % e.word
... 
14


#Get vectors for given word or index
>>> try:
...     print wv.get_word_vector(10, normalized=False)
... except UnknownIndex, e:
...     print "Index %d was not found" % e.index
...
[-3.2972147464752197 0.039462678134441376 0.7405596971511841
 5.008091926574707 0.8998156785964966]

>>> try:
...     print wv.get_word_vector(10, normalized=True)
... except UnknownIndex, e:
...     print "Index %d was not found" % e.index
... 
[-0.53978574  0.00646042  0.12123673  0.8198728   0.14730847]



#Get Word using Index
>>> try:
...     print wv.get_word_at_index(10)
... except UnknownIndex, e:
...     print "Index %d was not in the range" % e.index
... 
two



#Get occurrences of word
>>> print wv.get_word_occurrences(5327)
297

>>> try:
...     print wv.get_word_occurrences("to")
... except UnknownWord, e:
...     print "Word %s was not found" % e.word
316376


#Get Vector magnitude
>>> print wv.get_vector_magnitudes(["hi", 500])
[ 8.79479218  8.47650623]

>>> print wv.get_vector_magnitudes(["hfjsjfi", 500])
[ 0.          8.47650623]


#Get vectors for list of words
>>> print wv.get_word_vectors(["hi", "india"])
[[ 0.24728754  0.25350514 -0.32058391  0.80575693  0.35009396]
 [-0.62585545 -0.20999533  0.55592233 -0.36636305  0.34775764]]


#Get distance between two words using word or index
>>> print wv.get_distance(250, 500)
-0.288146

>>> print wv.get_distance(250, "india")
-0.163976


#Get distance between list of words
>>> print wv.get_distances("for", ["to", "for", "india"])
[[ 0.85009682]
 [ 1.00000012]
 [-0.38545406]]


>>> print wv.get_distances(["india", "for"], ["to", "for", "usa"])
[[-0.18296985 -0.38545409  0.51620466]
 [ 0.85009682  1.00000012 -0.49754807]]


#Get distance between list of words with all words in the word vector space
>>> print wv.get_distances(["india", "usa"])
[[-0.49026281  0.57980162  0.73099834 ..., -0.20406421 -0.35388517
   0.38457203]
 [-0.80836529  0.04589185 -0.16784868 ...,  0.4037039  -0.04579565
  -0.16079855]]


>>> print wv.get_distances(["andhra"])
[[-0.3432439   0.42185491  0.76944059 ..., -0.09365848 -0.13691582
   0.57156253]]


#Get nearest neighbours for given word or index
>>> print wv.get_nearest_neighbors(374, 20)
Int64Index([  374, 19146, 45990, 61134,  7975, 15522, 42578, 37966,  5326, 11644, 46233, 12635, 30945, 57543, 12802, 30845,  4601,  5847, 23795, 24323], dtype='int64')

 ```
