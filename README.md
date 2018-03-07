# WordVecSpace
A high performance pure python module that helps in loading and performing operations on word vector spaces created using Google's Word2vec tool.

This module has ability to load data into memory using `WordVecSpaceMem` or it can also supports performing operations on the data which is on the disk using `WordVecSpaceAnnoy`.

## Installation
> Prerequisites: Python3.5

```bash
$ sudo apt install libopenblas-base
$ sudo apt-get install libffi-dev
$ sudo pip3 install wordvecspace
```
> Note: wordvecspace is using `/usr/lib/libopenblas.so.0` as a default path for openblas. If the path of openblas is different in your machine then you have to set the environment variable for that path.
> Ex: For ubuntu-17.04, open blas path is `/usr/lib/x86_64-linux-gnu/libopenblas.so.0`.

Setting the environment variable for openblas
```bash
$ export WORDVECSPACE_BLAS_FPATH=/usr/lib/x86_64-linux-gnu/libopenblas.so.0
```

## Usage

### Preparing data

Before we can start using the library, we need access to some
word vector space data. Here are two ways to get that.

#### Download pre-computed sample data

```bash
$ wget https://s3.amazonaws.com/deepcompute-public-data/wordvecspace/small_test_data.tgz
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
# 3. Edit the command "time ./word2vec -train text8 -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15" ----to----> "time ./word2vec -train text8 -output vectors.bin -cbow 1 -size 5 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -save-vocab vocab.txt -iter 15" to get vocab.txt file also as output.
# 4. Run demo-word.sh

$ chmod +x demo-word.sh
$ ./demo-word.sh

# This will produce the output files (vectors.bin and vocab.txt)
```

These files (vectors.bin and vocab.txt) cannot be directly loaded
by the `wordvecspace` module. You'll first have to convert them
to the `WordVecSpace` format.

```bash
$ wordvecspace convert <input_dir> <output_file>

# <input_dir> is the directory which has vocab.txt and vectors.bin
# <output_file> is the file where you want to put your output file
```

Example:

```bash
$ wordvecspace convert /home/user/bindata /home/user/dc.wvspace

# /home/user/bindata is the directory containing vocab.txt and vectors.bin
# dc.wvspace is the output file
```

### Importing

#### Quick example

##### Import
```python
>>> from wordvecspace import WordVecSpaceMem
```

##### Load data
```python
>>> wv = WordVecSpaceMem('/home/user/dc.wvspace')
```

##### Make get_nearest call
```python
>>> wv.get_nearest('india', k=20)
[509, 486, 523, 4343, 14208, 13942, 42424, 25578, 6212, 2475, 3560, 13508, 20919, 3389, 4484, 19995, 8776, 7012, 12191, 16619]

# k is for getting top k nearest values
```

#### Types
`wordvecspace` module can perform operations by loading data into RAM using `WordVecSpaceMem` or directly on the data which is on the disk using `WordVecSpaceAnnoy`

`WordVecSpaceMem` is a bruteforce algorithm which compares given word with all the words in the vector space

`WordVecSpaceAnnoy` takes wvspace file as input and creates annoy indexes in another file (index file). Using this file `annoy` gives approximate results quickly. For better understanding of `Annoy` please go through this [link](https://github.com/spotify/annoy)

As we have seen how to import `WordVecSpaceMem` above, let us look at `WordVecSpaceAnnoy`

##### Import
```python
>>> from wordvecspace import WordVecSpaceAnnoy
```

##### Load data
```python
wv = WordVecSpaceAnnoy('/home/user/dc.wvspace', n_trees, index_fpath)

# n_trees = number of trees(More trees gives a higher precision when querying for get_nearest)
# index_fpath = path for annoy index file

# n_trees and index_fpath are optional. If those are not given then WordVecSpaceAnnoy uses `1` for n_trees and `/home/user/` (dc.wvspace file directory) directory for index_fpath.
```

##### Make get_nearest call
```python
>>> wv.get_nearest('india', k=20)
[509, 486, 523, 4343, 14208, 13942, 42424, 25578, 6212, 2475, 3560, 13508, 20919, 3389, 4484, 19995, 8776, 7012, 12191, 16619]
```

#### Distance calculations
`WordVecSpaceAnnoy` supports different types of distance calculations such as `"angular"`, `"euclidean"`, `"manhattan"` and `"hamming"`.

`WordVecSpaceMem` supports `"angular"` and `"euclidean"` for distance calculations.

Both uses `"angular"` by default. If you want to change it then you can change at the time of creating object.

Example:

```bash
wv = WordVecSpaceAnnoy('/path/to/wvspacefile', n_trees, metric="euclidean")
wv = WordVecSpaceMem('/path/to/wvspacefile', metric="euclidean")

# metric = type of distance calculation
```

WordVecSpaceMem can also supports specifying metric at the time of calculating distance.

Example:
```bash
wv = WordVecSpaceMem('/path/to/wvspacefile', metric="euclidean")

wv.get_distance('ap', 'india', metric='angular')
```

#### Examples of using wordvecspace methods

> WordVecSpaceMem and WordVecSpaceAnnoy have the same common methods.

##### Check if a word exists or not in the word vector space
```python
>>> print(wv.does_word_exist("india"))
True

>>> print(wv.does_word_exist("inidia"))
False
```

##### Get the index of a word
```python
>>> print(wv.get_word_index("india"))
509

>>> print(wv.get_word_index("inidia"))
None

>>> print(wv.get_word_index("inidia", raise_exc=True))
Traceback (most recent call last):
  File "/usr/lib/python3.6/code.py", line 91, in runcode
    exec(code, self.locals)
  File "<console>", line 1, in <module>
  File "/usr/local/lib/python3.6/dist-packages/wordvecspace/mem.py", line 196, in get_word_index
    raise UnknownWord(word)
wordvecspace.exception.UnknownWord: "inidia"
```

##### Get the indices of words
```python
>>> print(wv.get_word_indices(['the', 'deepcompute', 'india']))
[1, None, 509]

>>> print(wv.get_word_indices(['the', 'deepcompute', 'india'], raise_exc=True))
Traceback (most recent call last):
  File "/usr/lib/python3.6/code.py", line 91, in runcode
    exec(code, self.locals)
  File "<console>", line 1, in <module>
  File "/usr/local/lib/python3.6/dist-packages/wordvecspace/mem.py", line 209, in get_word_indices
    index = self.get_word_index(word, raise_exc=raise_exc)
  File "/usr/local/lib/python3.6/dist-packages/wordvecspace/mem.py", line 196, in get_word_index
    raise UnknownWord(word)
wordvecspace.exception.UnknownWord: "deepcompute"
```

##### Get Word at Index
```python
# Get word at Index 509
>>> print(wv.get_word_at_index(509))
india
```

##### Get Words at Indices
```python
>>> print(wv.get_word_at_indices([1, 509, 71190, 72000]))
['the', 'india', 'reka', None]
```

##### Get occurrence of the word
```python
# Get occurrences of the word "india"
>>> print(wv.get_word_occurrence("india"))
3242

# Get occurrences of the word "inidia"
>>> print(wv.get_word_occurrence("inidia"))
None
```

##### Get occurrence of the words
```python
# Get occurrence of the words 'the', 'india' and 'Deepcompute'
>>> print(wv.get_word_occurrences(["the", "india", "Deepcompute"]))
[1061396, 3242, None]
```

##### Get vector magnitude of the word
```python
# Get magnitude for the word "hi"
>>> print(wv.get_vector_magnitude("hi"))
8.7948
```

##### Get vector magnitude of the words
```python
# Get magnitude for the words "hi" and "india"
>>> print(wv.get_vector_magnitudes(["hi", "india"]))
[  8.7948  10.303 ]
```

##### Get vector for given word
```python
# Get the word vector for a word india
>>> print(wv.get_word_vector("india"))
[-6.4482 -2.1636  5.7277 -3.7746  3.583 ]

# Get the unit word vector for a word india
>>> print(wv.get_word_vector("india", normalized=True))
[-0.6259 -0.21    0.5559 -0.3664  0.3478]

# Get the word vector for a word inidia.
>>> print(wv.get_word_vector('inidia', raise_exc=True))
Traceback (most recent call last):
  File "/usr/lib/python3.6/code.py", line 91, in runcode
    exec(code, self.locals)
  File "<console>", line 1, in <module>
  File "/usr/local/lib/python3.6/dist-packages/wordvecspace/mem.py", line 287, in get_word_vector
    index = self.get_word_index(word, raise_exc)
  File "/usr/local/lib/python3.6/dist-packages/wordvecspace/mem.py", line 196, in get_word_index
    raise UnknownWord(word)
wordvecspace.exception.UnknownWord: "inidia"

# If you don't want to get exception when word is not there, then you can simply discard raise_exc=True
>>> print(wv.get_word_vector('inidia'))
[ 0.  0.  0.  0.  0.]
```

##### Get vector for given words
```python
>>> print(wv.get_word_vectors(["hi", "india"]))
[[ 0.4008  0.3623 -0.013   0.8395  0.0562]
 [-0.4975 -0.134   0.7874 -0.3274  0.0857]]
>>> print(wv.get_word_vectors(["hi", "inidia"]))
[[ 0.4008  0.3623 -0.013   0.8395  0.0562]
 [ 0.      0.      0.      0.      0.    ]]
```

##### Get distance between two words
```python
# Get distance between "india", "usa"
>>> print(wv.get_distance("india", "usa"))
0.48379534483

# Get the distance between 250, "india"
>>> print(wv.get_distance(250, "india"))
1.16397565603

# Get the euclidean distance between 250, "india" for WordvecSpaceMem
>>> print(wv.get_distance(250, "india", metric='euclidean'))
12.04961109161377
```

##### Get distance between list of words

```python
>>> print(wv.get_distances("for", ["to", "for", "india"]))
[[ 0.381   0.      0.9561]]

>>> print(wv.get_distances("for", ["to", "for", "inidia"]))
[[ 0.381  0.     1.   ]]

>>> print(wv.get_distances(["india", "for"], ["to", "for", "usa"]))
[[ 1.0685  0.9561  0.3251]
 [ 0.381   0.      1.4781]]

>>> print(wv.get_distances(["india", "usa"]))
[[ 1.3853  0.4129  0.3149 ...,  1.1231  1.4595  0.7912]
 [ 1.3742  0.9549  1.0354 ...,  0.5556  1.0847  1.0832]]

>>> print(wv.get_distances(["andhra"]))
[[ 1.2817  0.6138  0.2995 ...,  0.9945  1.224   0.6137]]

# For WordVecSpaceMem
>>> print(wv.get_distances(["andhra"], metric='euclidean'))
[[ 9.0035  8.3985  7.1658 ...,  9.2236  9.6078  8.6349]]
```

##### Get nearest
```python
# Get nearest for given word or index
>>> print(wv.get_nearest("india", 20))
[509, 486, 523, 4343, 14208, 13942, 42424, 25578, 6212, 2475, 3560, 13508, 20919, 3389, 4484, 19995, 8776, 7012, 12191, 16619]

# Get nearest for given words or indices
>>> print(wv.get_nearest(["ram", "india"], 5))
[[3844, 38851, 25381, 10830, 17049], [509, 486, 523, 4343, 14208]]

# Get nearest using euclidean distance for WordVecSpaceMem
>>> print(wv.get_nearest(["ram", "india"], 5, metric='euclidean'))
[[3844, 25381, 27802, 17049, 38851], [509, 486, 14208, 523, 13942]]

# Get common nearest neighbors among given words
>>> print(wv.get_nearest(['india', 'bosnia'], 10, combination=True))
[14208, 486, 523, 4343, 42424, 509]
```

### Service

```bash
# Run wordvecspace as a service (which continuously listens on some port for API requests)
$ wordvecspace runserver <type> <input_file> --metric <metric> --port <port> --eargs <eargs>

# <type> is for specifying wordvecspace functionality (eg: mem or annoy).
# <input_file> is for wordvecspace file
# <metric> is to specify type for distance calculation
# <port> is to run wordvecspace in that port
# <eargs> is for specifying extra arguments for annoy
```

Example:

```bash
# For mem
$ wordvecspace runserver mem /home/user/dc.wvspace --metric angular --port 8000

# For annoy
$ wordvecspace runserver annoy /home/user/dc.wvspace --metric euclidean --port 8000 --eargs n_trees=1:index_fpath=/tmp

# Extra arguments for annoy are n_trees and index_fpath
#   - n_trees is the number of trees for annoy
#   - index_fpath is the directory for annoy index file

# Make API request
$ curl "http://localhost:8000/api/v1/does_word_exist?word=india"
{"result": true, "success": true}
```

#### Making call to all API methods

```bash
$ http://localhost:8000/api/v1/does_word_exist?word=india

$ http://localhost:8000/api/v1/get_word_index?word=india

$ http://localhost:8000/api/v1/get_word_indices?words=["india", 22, "hello"]

$ http://localhost:8000/api/v1/get_word_at_index?index=509

$ http://localhost:8000/api/v1/get_word_at_indices?indices=[22, 509]

$ http://localhost:8000/api/v1/get_word_vector?word_or_index=509

$ http://localhost:8000/api/v1/get_vector_magnitude?word_or_index=88

$ http://localhost:8000/api/v1/get_vector_magnitudes?words_or_indices=[88, "india"]

$ http://localhost:8000/api/v1/get_word_occurrence?word_or_index=india

$ http://localhost:8000/api/v1/get_word_occurrences?words_or_indices=["india", 22]

$ http://localhost:8000/api/v1/get_word_vectors?words_or_indices=[1, "india"]

$ http://localhost:8000/api/v1/get_distance?word_or_index1=ap&word_or_index2=india

$ http://localhost:8000/api/v1/get_distances?row_words_or_indices=["india", 33]

$ http://localhost:8000/api/v1/get_nearest?words_or_indices=india&k=100

$ http://localhost:8000/api/v1/get_nearest?words_or_indices=india&k=100&metric=euclidean
```

> To see all API methods of wordvecspace please run http://localhost:8000/api/v1/apidoc

### Interactive console
```bash
# wordvecspace provides command to directly interact with it

$ wordvecspace interact <type> <input_file> --metric <metric> --eargs <eargs>

# <type> is for specifying wordvecspace functionality (eg: mem or annoy).
# <input_file> is for wordvecspace file
# <metric> is to specify type for distance calculation
# <eargs> is for specifying extra arguments for annoy
```

Example:
```bash
# For mem
$ wordvecspace interact mem /home/user/dc.wvspace --metric euclidean

$ wordvecspace interact annoy /home/user/dc.wvspace --metric angular --eargs n_trees=1:index_fpath=/tmp
WordVecSpaceAnnoy console (vectors=71291 dims=5)
>>> wv.get_nearest('india', 20)
[509, 486, 523, 4343, 13942, 42424, 25578, 3389, 12191, 16619, 12088, 6049, 5226, 4137, 41883, 18617, 10172, 35704, 25552, 29059]

# Extra arguments for annoy are n_trees and index_fpath
#   - n_trees is the number of trees for annoy
#   - index_fpath is the directory for annoy index file
```

## Running tests
```bash
# Download the data files
$ wget 'https://s3.amazonaws.com/deepcompute-public-data/wordvecspace/small_test_data.tgz'

# Extract downloaded small_test_data.tgz file
$ tar xvzf small_test_data.tgz

# Export the path of data file to the environment variables
$ export WORDVECSPACE_DATAFILE="/home/user/dc.wvspace"

# Run tests
$ python3 setup.py test
```

## GPU acceleration

`wordvecspace` can take advantage of an Nvidia GPU to perform some operations significantly faster. This is as simple as doing

```python
>>> from wordvecspace.cuda import WordVecSpaceMem
```

The `WordVecSpaceMem` from the `cuda` module is a drop-in replacement for the CPU based `WordVecSpaceMem` class showcased above.

> NOTE: The vector space size must fit on available GPU RAM for this to work
> Also, you will need to install cuda support by doing "sudo pip3 install wordvecspace[cuda]"
