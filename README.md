# Word Vector Space Abstraction

## Installation


Install python-2.7

sudo pip install numpy pandas numba
sudo apt install libopenblas-dev

### Usage of wordvecspace library

$python

#Import classes from wordvecspace library
>>>from wordvecspace import WordVecSpace, UnknownWord, UnknownIndex

#Specify the path of vectors.npy and VOCAB.txt
>>>data_dir = "/home/deepcompute/alpha/data/w2v_new_sharded/shard_0"

#Create Instace for WordVecSpace
>>>wv = WordVecSpace(data_dir)


#Load the data files into memory
>>>wv.load()

#Check word exist or not in wordvecspce
>>>print wv.does_word_exist("imatinib")
True

>>>print wv.does_word_exist("imatislkus")
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
2

#Get vectors for given word or index
>>> try:
...     print wv.get_word_vector(10, normalized=False)
... except UnknownIndex, e:
...     print "Index %d was not found" % e.index
... 
[ 0.01522985  0.00267594 .... -0.00906473]

#Get normalized vectors for given word or index
>>> try:
...     print wv.get_word_vector(10, normalized=True)
... except UnknownIndex, e:
...     print "Index %d was not found" % e.index
... 
[0.08531315624713898 0.014989836141467094 .... -0.05077797546982765]



#Get Word using Index
>>> try:
...     print wv.get_word_at_index(10)
... except UnknownIndex, e:
...     print "Index %d was not in the range" % e.index
... 
pubmed



#Get occurrences of word
>>> print wv.get_word_occurrences(5327)
664333

>>> try:
...     print wv.get_word_occurrences("to")
... except UnknownWord, e:
...     print "Word %s was not found" % e.word
616385965


#Get Vectors magnitude
>>> print wv.get_vector_magnitudes(["hi", 500])
[5.1710873, 3.8780088]

>>> print wv.get_vector_magnitudes(["hfjsjfi", 500])
[0.0 3.87800884]


#Get vectors for list of words
>>> print wv.get_word_vectors(["hi", "imatinib"])
[[  4.58009765e-02   2.27097664e-02 ... -4.50771116e-02]
 [  2.15231422e-02   7.32142106e-02 ... -7.41100591e-03]]

#Get distance between two words using word or index
>>> print wv.get_distance("250", "500")
0.817561

>>> print wv.get_distance("250", "imatinib")
0.13943

#Get distance between list of words
>>> print wv.get_distances("for", ["to", "for", "imatinib"])
[[ 0.80703819]
 [ 0.99999988]
 [ 0.27108291]]

>>> print wv.get_distances(["nilotinib", "for"], ["to", "for", "imatinib"])
[[ 0.23537777  0.20905481  0.88973904]
 [ 0.80703843  0.99999982  0.27108291]]

#Get distance between list of words with all words in the word vector space
>>> print wv.get_distances(["nilotinib", "hi"])
[[ 0.0601333   0.23537777  0.20905481 ...,  0.22716512  0.2496517
   0.25396603]
 [ 0.05879326  0.36978272  0.35755485 ...,  0.21065465  0.21103515  0.19593   ]]


>>> print wv.get_distances(["imatinib"])
[[ 0.03310118  0.27105609  0.27108291 ...,  0.25952423  0.22930798
   0.22244862]]


