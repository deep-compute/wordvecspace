
# word vector space
A python module that helps in loading and performing operations on word vector spaces created using Google's Word2vec tool over GPU and  CPU

## Installation
> Prerequisites: Python2.7

```bash=!
sudo pip install numpy pandas numba pycuda
git clone "https://github.com/lebedov/scikit-cuda.git"
sudo apt install libopenblas-base
```

## Usage

```python=!
# GPU
>>> from wordvecspace.cuda import WordVecSpace

# CPU
>>> from wordvecspace import WordVecSpace

# Load the data (Vector and Vocab information)
>>> wv = WordVecSpace('/path/to/data/')
>>> wv.load()

# Check if a word exists or not in the word vector space
>>> print wv.does_word_exist("india")
True

>>> print wv.does_word_exist("inidia")
False

# Get the index of a word

>>> print wv.get_word_index("india", raise_exc=False)
509
>>> print wv.get_word_index("india")
509

>>> print wv.get_word_index("inidia", raise_exc=True)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>

  File "wordvecspace.py", line 186, in get_word_index

    raise UnknownWord(word)
wordvecspace.UnknownWord: "inidia"

# Get vector for given word or index

# Get the word vector for a word india

>>> print wv.get_word_vector("india", raise_exc=False)
[-6.44819975 -2.16358566  5.72767735 -3.77464485  3.58295298]

# Get the unit word vector for a word india
>>> print wv.get_word_vector("india", normalized=True, raise_exc=False)
[-0.62585545 -0.20999533  0.55592233 -0.36636305  0.34775764]

>>> print wv.get_word_vector("india")
[-6.44819975 -2.16358566  5.72767735 -3.77464485  3.58295298]

# Get the unit word vector for a word india
>>> print wv.get_word_vector("india", normalized=True)
[-0.62585545 -0.20999533  0.55592233 -0.36636305  0.34775764]

# Get the unit vector for a word inidia.
>>> print wv.get_word_vector('inidia', normalized=True, raise_exc=True)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "wordvecspace.py", line 219, in get_word_vector
    index = self.get_word_index(word_or_index, raise_exc)
  File "wordvecspace.py", line 184, in get_word_index
    raise UnknownWord(word)
wordvecspace.UnknownWord: "inidia"

# Get the unit vector for a word inidia. If the word is not present it simply returns zeros if raise_exc is False.
>>> print wv.get_word_vector('inidia', normalized=True, raise_exc=False)
[ 0.  0.  0.  0.  0.]

# Get Word at Index 509
>>> print wv.get_word_at_index(509, raise_exc=False)
india

# Get occurrences of the word "india"
>>> print wv.get_word_occurrences("india", raise_exc=False)
3242

# Get occurrences of the word "to"
>>> print wv.get_word_occurrences("to", raise_exc=False)
316376

# Get occurrences of the word inidia
>>> print wv.get_word_occurrences("inidia", raise_exc=False)
None

# Get Vector magnitude of the word india
>>> print wv.get_vector_magnitudes("india", raise_exc=False)
[ 10.30301762]
# In above, you can alternatively pass the index of "india" instead of the word itself

>>> print wv.get_vector_magnitudes(["india", "usa"], raise_exc=False)
[ 10.30301762   7.36207819]

>>> print wv.get_vector_magnitudes(["inidia", "usa"], raise_exc=False)
[ 0.          7.36207819]

# Get vectors for list of words

>>> print wv.get_word_vector('inidia', normalized=True)
[ 0.  0.  0.  0.  0.]

# Get Word at Index 509
>>> print wv.get_word_at_index(509)
india

# Get occurrences of the word india
>>> print wv.get_word_occurrences("india")
3242

# Get occurrences of the word to
>>> print wv.get_word_occurrences("to")
316376

# Get occurrences of the word inidia
>>> print wv.get_word_occurrences("inidia")
None

# Get occurrences of the word inidia
>>> print wv.get_word_occurrences("inidia", raise_exc=True)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "wordvecspace.py", line 268, in get_word_occurrences
    index = self.get_word_index(word_or_index, raise_exc)
  File "wordvecspace.py", line 184, in get_word_index
    raise UnknownWord(word)
wordvecspace.UnknownWord: "inidia"

# Get Vector magnitude of the word india
>>> print wv.get_vector_magnitudes("india")
[ 10.30301762]
# In above, you can alternatively pass the index of "india" instead of the word itself

>>> print wv.get_vector_magnitudes(["india", "usa"])
[ 10.30301762   7.36207819]

>>> print wv.get_vector_magnitudes(["inidia", "usa"])
[ 0.          7.36207819]
	
# Get vectors for list of words
>>> print wv.get_word_vectors(["usa", "india"])
[[-0.72164571 -0.05566886  0.41082662  0.54941767  0.07409521]
 [-0.62585545 -0.20999533  0.55592233 -0.36636305  0.34775764]]

# Get distance between two words using word or index

>>> print wv.get_distance("india", "usa", raise_exc=False)
0.516205

>>> print wv.get_distance(250, "india", raise_exc=False)
-0.163976

# Get distance between list of words
>>> print wv.get_distances("for", ["to", "for", "india"], raise_exc=False)
[[ 0.85009682]
 [ 1.00000012]
 [-0.38545406]]

>>> print wv.get_distances("for", ["to", "for", "inidia"], raise_exc=False)
[[ 0.85009682]
 [ 1.00000012]
 [ 0.        ]]

>>> print wv.get_distances(["india", "for"], ["to", "for", "usa"], raise_exc=False)
[[-0.18296985 -0.38545409  0.51620466]
 [ 0.85009682  1.00000012 -0.49754807]]

# Get distance between list of words with all words in the word vector space
>>> print wv.get_distances(["india", "usa"], raise_exc=False)
[[-0.49026281  0.57980162  0.73099834 ..., -0.20406421 -0.35388517
   0.38457203]
 [-0.80836529  0.04589185 -0.16784868 ...,  0.4037039  -0.04579565
  -0.16079855]]

>>> print wv.get_distances(["andhra"], raise_exc=False)
[[-0.3432439   0.42185491  0.76944059 ..., -0.09365848 -0.13691582
   0.57156253]]

# Get nearest neighbours for given word or index
>>> print wv.get_nearest_neighbors("india", 20)
...

>>> print wv.get_distance("india", "usa")
0.48379534483

# Get distance between list of words
>>> print wv.get_distances("for", ["to", "for", "india"])
[[  1.49903178e-01]
 [ -1.19209290e-07]
 [  1.38545406e+00]]

>>> print wv.get_distances("for", ["to", "for", "inidia"])
[[  1.49903178e-01]
 [ -1.19209290e-07]
 [  1.00000000e+00]]

>>> print wv.get_distances(["india", "for"], ["to", "for", "usa"])
[[  1.18296981e+00   1.38545406e+00   4.83795345e-01]
 [  1.49903178e-01  -1.19209290e-07   1.49754810e+00]]

# Get distance between list of words with all words in the word vector space
>>> print wv.get_distances(["india", "usa"])
[[ 1.49026275  0.42019838  0.26900166 ...,  1.20406425  1.35388517
   0.61542797]
 [ 1.80836535  0.95410818  1.16784871 ...,  0.59629607  1.04579568
   1.16079855]]

>>> print wv.get_distances(["andhra"])
[[ 1.34324384  0.57814509  0.23055941 ...,  1.09365845  1.1369158
   0.42843747]]

# Get nearest neighbours for given word or index
>>> print wv.get_nearest_neighbors("india", 20)
... 

Int64Index([  509,   486, 14208, 20639,  8573,  3389,  5226, 20919, 10172,
             6866,  9772, 24149, 13942,  1980, 20932, 28413, 17910,  2196,
            28738, 20855],
           dtype='int64')
```

## Running tests

```bash=!
# Download the data files
$ wget 'https://s3.amazonaws.com/deepcompute-public/data/wordvecspace/small_test_data.tgz'

# Extract downloaded small_test_data.tgz file
$ tar xvzf small_test_data.tgz

# Export the path of data files to the environment variables
$ export WORDVECSPACE_DATADIR="/home/ram/small_test_data"

# Run doctest
python -m doctest -v wordvecspace.py

```
