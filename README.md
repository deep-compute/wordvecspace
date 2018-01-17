
# WordVecSpace
A high performance pure python module that helps in loading and performing operations on word vector spaces created using Google's Word2vec tool.

## Installation
> Prerequisites: Python2.7

```bash
$ sudo apt install libopenblas-base
$ sudo pip install wordvecspace
```

## Usage

### Preparing data

Before we can start using the library, we need access to some
word vector space data. Here are two ways to get that.

#### Download pre-computed sample data

```bash
$ wget https://s3.amazonaws.com/deepcompute-public/data/wordvecspace/small_test_data.tgz
$ tar zxvf small_test_data.tgz
```

> NOTE: We got this data by downloading the `text8` corpus
> from this location (http://mattmahoney.net/dc/text8.zip) and converting that to `WordVecSpace`
> format. You can do the same conversion process by reading
> the instructions in the following section.

#### Computing your own data

You can compute a word vector space on an arbitrary text corpus
by using Google's word2vec tool. Here is an example on how to do
that for the sample `text8` corpus.

```bash
$ git clone https://github.com/tmikolov/word2vec.git 

# 1. Navigate to the folder word2vec
# 2. open demo-word.sh for editing
# 3. Edit "time ./word2vec -train text8 -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15" to "time ./word2vec -train text8 -output vectors.bin -cbow 1 -size 5 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -save-vocab vocab.txt -iter 15" to get vocab.txt file also as output.
# 4. Run demo-word.sh

$ chmod +x demo-word.sh
$ ./demo-word.sh

# This will produce the output files (vectors.bin and vocab.txt)
```

These files (vectors.bin and vocab.txt) cannot be directly loaded
by the `wordvecspace` module. You'll first have to convert them
to the `WordVecSpace` format.


```bash
$ wordvecspace convert <input_dir> <output_dir>

# <input_dir> is the directory which has vocab.txt and vectors.bin
# <output_dir> is the directory where you want to put your output files

# You can also generate shards by specifying number of vectors per each shard
$ wordvecspace convert <input_dir> <output_dir> -n 5000
```
### wordvecspace can be used in two ways (Interact or Import)
#### Interact
```bash
$ wordvecspace interact <input_dir>
```

#### Import and load data (Vector and Vocab information)
```python
>>> from wordvecspace import WordVecSpace
>>> wv = WordVecSpace('/path/to/data/')
>>> wv.load()
```

### Check if a word exists or not in the word vector space
```python
>>> print wv.does_word_exist("india")
True

>>> print wv.does_word_exist("inidia")
False
```

### Get the index of a word
```python
>>> print wv.get_word_index("india")
509
>>> print wv.get_word_index("india")
509

>>> print wv.get_word_index("inidia", raise_exc=True)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>

  File "wordvecspace.py", line 186, in get_word_index

    raise UnknownWord(word)
wordvecspace.UnknownWord: "inidia"
```

### Get vector for given word or index
```python
# Get the word vector for a word india

>>> print wv.get_word_vector("india")
[-6.4482 -2.1636  5.7277 -3.7746  3.583 ]

# Get the unit word vector for a word india
>>> print wv.get_word_vector("india", normalized=True)
[-0.6259 -0.21    0.5559 -0.3664  0.3478]

>>> print wv.get_word_vector("india")
[-6.4482 -2.1636  5.7277 -3.7746  3.583 ]

# Get the unit word vector for a word india
>>> print wv.get_word_vector("india", normalized=True)
[-0.6259 -0.21    0.5559 -0.3664  0.3478]

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
>>> print wv.get_word_vector('inidia', normalized=True)
[ 0.  0.  0.  0.  0.]
```

### Get Word at Index 
```python
# Get word at Index 509
>>> print wv.get_word_at_index(509)
india
```
### Get occurrences of the word 
```python
# Get occurrences of the word "india"
>>> print wv.get_word_occurrences("india")
3242

# Get occurrences of the word "to"
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
```

### Get Vector magnitude of the word 
```python
# Get Vector magnitude of the word "india"
>>> print wv.get_vector_magnitudes("india")
[ 10.303]

>>> print wv.get_vector_magnitudes(["india", "usa"])
[ 10.303   7.3621]

>>> print wv.get_vector_magnitudes(["inidia", "usa"])
[ 0.          7.3621]

>>> print wv.get_vector_magnitudes(["india", "usa"])
[ 10.303    7.3621]

>>> print wv.get_vector_magnitudes(["inidia", "usa"])
[ 0.          7.3621]
```	

### Get vectors for list of words
```python
# Get vectors for list of words ["usa", "india"]
>>> print wv.get_word_vectors(["usa", "india"])
[[-0.7216 -0.0557  0.4108  0.5494  0.0741]
 [-0.6259 -0.21    0.5559 -0.3664  0.3478]]
```

### Get distance between two words 
```python
# Get distance between "india", "usa"
>>> print wv.get_distance("india", "usa")
0.48379534483

# Get the distance between 250, "india"
>>> print wv.get_distance(250, "india")
1.16397565603
```

### Get distance between list of words
```python
>>> print wv.get_distances("for", ["to", "for", "india"])
[[  1.4990e-01]
 [ -1.1921e-07]
 [  1.3855e+00]]

>>> print wv.get_distances("for", ["to", "for", "inidia"])
[[  1.4990e-01]
 [ -1.1921e-07]
 [  1.0000e+00]]

>>> print wv.get_distances(["india", "for"], ["to", "for", "usa"])
[[  1.1830e+00   1.3855e+00   4.8380e-01]
 [  1.4990e-01  -1.1921e-07   1.4975e+00]]

>>> print wv.get_distances(["india", "usa"])
[[ 1.4903  0.4202  0.269  ...,  1.2041  1.3539  0.6154]
 [ 1.8084  0.9541  1.1678 ...,  0.5963  1.0458  1.1608]]

>>> print wv.get_distances(["andhra"])
[[ 1.3432  0.5781  0.2306 ...,  1.0937  1.1369  0.4284]]
```

### Get nearest neighbors 
```python
# Get nearest neighbours for given word or index
>>> print wv.get_nearest_neighbors("india", 20)
... 

Int64Index([  509,   486, 14208, 20639,  8573,  3389,  5226, 20919, 10172,
             6866,  9772, 24149, 13942,  1980, 20932, 28413, 17910,  2196,
            28738, 20855],
           dtype='int64')
```

## Running tests

```bash
# Download the data files
$ wget 'https://s3.amazonaws.com/deepcompute-public/data/wordvecspace/small_test_data.tgz'

# Extract downloaded small_test_data.tgz file
$ tar xvzf small_test_data.tgz

# Export the path of data files to the environment variables
$ export WORDVECSPACE_DATADIR="/home/ram/small_test_data"

# Run tests
$ python setup.py test
```

## GPU acceleration

`wordvecspace` can take advantage of an Nvidia GPU to perform some operations significantly faster. This is as simple as doing
```bash
$ pip install wordvecspace[cuda]
```

```python
>>> from wordvecspace.cuda import WordVecSpace
```

The `WordVecSpace` from the `cuda` module is a drop-in replacement for the CPU based `WordVecSpace` class showcased above.

> NOTE: The vector space size must fit on available GPU ram for this to work
> Also, you will need to install cuda support by doing "sudo pip install wordvecspace[cuda]"
