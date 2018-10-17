# WordVecSpace
A high performance pure python module that helps in loading and performing operations on word vector spaces created using Google's Word2vec tool.

This module has ability to the load data into memory using `WordVecSpaceMem` and it can also support performing operations on the data which is on the disk using `WordVecSpaceAnnoy` and                   `WordVecSpaceDisk`.

## Installation
> Prerequisites: >=Python3.5.2

```bash

$ sudo apt install libopenblas-base # Optional
$ sudo pip3 install wordvecspace
```

## Usage

### Preparing data

Before we can start using the library, we need access to some
word vector space data. Here are two ways to get that.

#### Download pre-computed sample data

```bash
$ wget https://s3.amazonaws.com/deepcompute-public-data/wordvecspace/test_data-0_5_4.tgz
$ tar zxvf test_data-0_5_4.tgz
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
$ wordvecspace convert <input_dir> <output_dir>

# <input_dir> is the directory which has vocab.txt and vectors.bin
# <output_dir> is the directory where you want to store your output files.
```

Example:

```bash
$ wordvecspace convert /home/user/bindata /home/user/output_dir

# /home/user/bindata is the directory containing vocab.txt and vectors.bin
# /home/user/output_dir is the output directory which contains wordvecspace data files.
```

### Importing

#### Quick example

##### Import
```python
>>> from wordvecspace import WordVecSpaceDisk
```

##### Load data
```python
>>> wv = WordVecSpaceDisk('/home/user/output_dir')
```

##### Make get_nearest call
```python
>>> wv.get_nearest('india', k=20)
[509, 3389, 486, 523, 7125, 16619, 4491, 12191, 6866, 8776, 15232, 14208, 5998, 21916, 5226, 6322, 4343, 6212, 10172, 6186]
# k is for getting top k nearest values
```

#### Types
`wordvecspace` module can perform operations by loading data into RAM using `WordVecSpaceMem` or directly on the data which is on the disk using `WordVecSpaceDisk`

`WordVecSpaceMem` and `WordVecSpaceDisk` is a bruteforce algorithm which compares given word with all the words in the vector space

`WordVecSpaceAnnoy` takes wordvecspace output_dir as input and creates annoy indexes in another file (index file). Using this file `annoy` gives approximate results quickly. For better understanding of `Annoy` please go through this [link](https://github.com/spotify/annoy)

As we have seen how to import `WordVecSpaceDisk` above, let us look at `WordVecSpaceAnnoy` and `WordVecSpaceMem`

##### Import
```python
>>> from wordvecspace import WordVecSpaceAnnoy
>>> from wordvecspace import WordVecSpaceMem
```

##### Load data
```python
# WordVecSpaceMem
>>> wv = WordVecSpaceMem('/home/user/output_dir')

# WordVecSpaceAnnoy
>>> wv = WordVecSpaceAnnoy('/home/user/output_dir', n_trees=2, index_fpath='/tmp')

# n_trees = number of trees(More trees gives a higher precision when querying for get_nearest)
# index_fpath = path for annoy index file

# n_trees and index_fpath are optional. If those are not given then WordVecSpaceAnnoy uses `1` for n_trees and `/home/user/output_dir` (wordvecspace data directory) directory for index_fpath.
```

##### Make get_nearest call
```python
>>> wv.get_nearest('india', k=20) (MEM)
[509, 3389, 486, 523, 7125, 16619, 4491, 12191, 6866, 8776, 15232, 14208, 5998, 21916, 5226, 6322, 4343, 6212, 10172, 6186]

>>> wv.get_nearest('india', k=20) (ANNOY)
[509, 3389, 16619, 4491, 6866, 8776, 14208, 5998, 21916, 20919, 2325, 4622, 3546, 24149, 5064, 35704, 25578, 15842, 4137, 6499]
```

#### Distance calculations
`WordVecSpaceAnnoy` supports different types of distance calculations such as `"angular"`, `"euclidean"`, `"manhattan"` and `"hamming"`.

`WordVecSpaceMem` supports `"angular"` and `"euclidean"` for distance calculations.

`WordVecSpaceDisk` supports `"angular"` and `"euclidean"` for distance calculations.

All of the above uses `"angular"` by default. If you want to change it then you can change at the time of creating object.

Example:

```bash
wv = WordVecSpaceAnnoy('/path/to/output_dir', n_trees, metric="euclidean")
wv = WordVecSpaceMem('/path/to/output_dir', metric="euclidean")
wv = WordVecSpaceDisk('/path/to/output_dir', metric="euclidean")

# metric = type of distance calculation
```

WordVecSpaceMem can also supports specifying metric at the time of calculating distance.

Example:
```bash
wv = WordVecSpaceMem('/path/to/output_dir', metric="euclidean")

wv.get_distance('ap', 'india', metric='angular')
```

#### Examples of using wordvecspace methods

> `WordVecSpaceMem`, `WordVecSpaceAnnoy` and `WordVecSpaceDisk` support the same methods.

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

>>> print(wv.get_index("inidia"))
None

>>> print(wv.get_index("inidia", raise_exc=True))
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
>>> print(wv.get_indices(['the', 'deepcompute', 'india']))
[1, None, 509]


>>> print(wv.get_indices(['the', 'deepcompute', 'india'], raise_exc=True))
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
>>> print(wv.get_word(509))
india
```

##### Get Words at Indices
```python
>>> print(wv.get_words([1, 509, 71190, 72000]))
['the', 'india', 'reka', None]
```

##### Get occurrence of the word
```python
# Get occurrences of the word "india"
>>> print(wv.get_occurrence("india"))
3242

# Get occurrences of the word "inidia"
>>> print(wv.get_occurrence("inidia"))
None
```

##### Get occurrence of the words
```python
# Get occurrence of the words 'the', 'india' and 'Deepcompute'
>>> print(wv.get_occurrences(["the", "india", "Deepcompute"]))
[1061396, 3242, None]
```

##### Get vector magnitude of the word
```python
# Get magnitude for the word "hi"
>>> print(wv.get_magnitude("hi"))
1.0
```

##### Get vector magnitude of the words
```python
# Get magnitude for the words "hi" and "india"
>>> print(wv.get_magnitudes(["hi", "india"]))
[1.0, 1.0]
```

##### Get vector for given word
```python
# Get the word vector for a word india
>>> print(wv.get_vector("india"))
[-0.7871 -0.2993  0.3233 -0.2864  0.323 ]

# Get the unit word vector for a word india
>>> print(wv.get_vector("india", normalized=True))
[-0.7871 -0.2993  0.3233 -0.2864  0.323 ]

# Get the word vector for a word inidia.
>>> print(wv.get_vector('inidia', raise_exc=True))
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
>>> print(wv.get_vector('inidia'))
[ 0.  0.  0.  0.  0.]
```

##### Get vector for given words
```python
>>> print(wv.get_vectors(["hi", "india"]))
[[ 0.6342  0.2268 -0.3904  0.0368  0.6266]
 [-0.7871 -0.2993  0.3233 -0.2864  0.323 ]]
>>> print(wv.get_vectors(["hi", "inidia"]))
[[ 0.6342  0.2268 -0.3904  0.0368  0.6266]
 [ 0.      0.      0.      0.      0.    ]]
```

##### Get distance between two words
```python
# Get distance between "india", "usa"
>>> print(wv.get_distance("india", "usa"))
0.37698328495

# Get the distance between 250, "india"
>>> print(wv.get_distance(250, "india"))
1.1418992728

# Get the euclidean distance between 250, "india" for WordvecSpaceMem
>>> print(wv.get_distance(250, "india", metric='euclidean'))
1.5112241506576538
```

##### Get distance between list of words

```python
>>> print(wv.get_distances("for", ["to", "for", "india"]))
[[  2.7428e-01   5.9605e-08   1.1567e+00]]

>>> print(wv.get_distances("for", ["to", "for", "inidia"]))
[[  2.7428e-01   5.9605e-08   1.0000e+00]]

>>> print(wv.get_distances(["india", "for"], ["to", "for", "usa"]))
[[  1.1445e+00   1.1567e+00   3.7698e-01]
 [  2.7428e-01   5.9605e-08   1.6128e+00]]

>>> print(wv.get_distances(["india", "usa"]))
[[ 1.5464  0.4876  0.3017 ...,  1.2492  1.2451  0.8925]
 [ 1.0436  0.9995  1.0913 ...,  0.6996  0.8014  1.1608]]

>>> print(wv.get_distances(["andhra"]))
[[ 1.5418  0.7153  0.277  ...,  1.1657  1.0774  0.7036]]

# For WordVecSpaceMem
>>> print(wv.get_distances(["andhra"], metric='euclidean'))
[[ 1.756   1.1961  0.7443 ...,  1.5269  1.4679  1.1862]]
```

##### Get nearest
```python
# Get nearest for given word or index
>>> print(wv.get_nearest("india", 20))
[509, 3389, 486, 523, 7125, 16619, 4491, 12191, 6866, 8776, 15232, 14208, 5998, 21916, 5226, 6322, 4343, 6212, 10172, 6186]

# Get nearest for given words or indices
>>> print(wv.get_nearest(["ram", "india"], 5))
[[3844, 16727, 15811, 42731, 41516], [509, 3389, 486, 523, 7125]]

# Get nearest using euclidean distance for WordVecSpaceMem
>>> print(wv.get_nearest(["ram", "india"], 5, metric='euclidean'))
[[3844, 16727, 15811, 42731, 41516], [509, 3389, 486, 523, 7125]]

# Get common nearest neighbors among given words
>>> wv.get_words(wv.get_nearest(['india', 'pakistan'], 10)[0])
['india', 'indian', 'delhi', 'subcontinent', 'hyderabad', 'pradesh', 'pakistan', 'gujarat', 'bombay', 'chhattisgarh']
>>> wv.get_words(wv.get_nearest(['india', 'pakistan'], 10)[1])
['pakistan', 'pakistani', 'india', 'bangladesh', 'peshawar', 'afghanistan', 'baluchistan', 'balochistan', 'kashmir', 'islamabad']
>>> wv.get_words(wv.get_nearest(['india', 'pakistan'], 10, combination=True)[0])
['pakistan', 'india', 'indian', 'bangladesh', 'pakistani', 'subcontinent', 'shimla', 'delhi', 'punjab', 'ladakh']
>>> wv.get_words(wv.get_nearest(['india', 'pakistan'], 10, combination=True, weights=[1, 0])[0])
['india', 'indian', 'delhi', 'subcontinent', 'hyderabad', 'pradesh', 'pakistan', 'gujarat', 'bombay', 'chhattisgarh']
>>> wv.get_words(wv.get_nearest(['india', 'pakistan'], 10, combination=True, weights=[0, 1])[0])
['pakistan', 'pakistani', 'india', 'bangladesh', 'peshawar', 'afghanistan', 'baluchistan', 'balochistan', 'kashmir', 'islamabad']
>>> wv.get_words(wv.get_nearest(['india', 'pakistan'], 10, combination=True, weights=[0.7, 0.3])[0])
['india', 'pakistan', 'indian', 'subcontinent', 'delhi', 'bangladesh', 'hyderabad', 'shimla', 'punjab', 'bengal']
>>> wv.get_words(wv.get_nearest(['india', 'pakistan'], 10, combination=True, weights=[0.3, 0.7])[0])
['pakistan', 'india', 'pakistani', 'bangladesh', 'subcontinent', 'indian', 'shimla', 'punjab', 'kashmir', 'ladakh']

# Get nearest with vector(s)
>>> wv.get_words(wv.get_nearest(wv.get_vector('india').reshape(1, wv.dim), k=5))
['india', 'indian', 'subcontinent', 'bombay', 'bengal']
>>> wv.get_words(wv.get_nearest(wv.get_vectors(['india', 'pakistan']), k=5)[0])
['india', 'indian', 'subcontinent', 'bombay', 'bengal']
>>> wv.get_words(wv.get_nearest(wv.get_vectors(['india', 'pakistan']), k=5)[1])
['pakistan', 'pakistani', 'kargil', 'afghanistan', 'bangladesh']
>>> wv.get_words(wv.get_nearest(wv.get_vectors(['india', 'pakistan']), k=5, combination=True)[0])
['india', 'pakistan', 'indian', 'pakistani', 'subcontinent']
>>> wv.get_words(wv.get_nearest(wv.get_vectors(['india', 'pakistan']), k=5, combination=True, weights=[0.4, 0.6])[0])
['pakistan', 'india', 'pakistani', 'kargil', 'indian']

```

## Service

```bash
# Run wordvecspace as a service (which continuously listens on some port for API requests)
$ wordvecspace runserver <type> <input_dir> --metric <metric> --port <port> --eargs <eargs>

# <type> is for specifying wordvecspace functionality (eg: mem, annoy or disk).
# <input_dir> is for wordvecspace data dir
# <metric> is to specify type for distance calculation
# <port> is to run wordvecspace in that port
# <eargs> is for specifying extra arguments for annoy
```

Example:

```bash
# For mem
$ wordvecspace runserver mem /home/user/output_dir --metric angular --port 8000

# For disk
$ wordvecspace runserver disk /home/user/output_dir --metric angular --port 8000

# For annoy
$ wordvecspace runserver annoy /home/user/output_dir --metric euclidean --port 8000 --eargs n_trees=1:index_fpath=/tmp

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

$ http://localhost:8000/api/v1/get_index?word=india

$ http://localhost:8000/api/v1/get_indices?words=["india", 22, "hello"]

$ http://localhost:8000/api/v1/get_index?index=509

$ http://localhost:8000/api/v1/get_indices?indices=[22, 509]

$ http://localhost:8000/api/v1/get_vector?word_or_index=509

$ http://localhost:8000/api/v1/get_magnitude?word_or_index=88

$ http://localhost:8000/api/v1/get_magnitudes?words_or_indices=[88, "india"]

$ http://localhost:8000/api/v1/get_occurrence?word_or_index=india

$ http://localhost:8000/api/v1/get_occurrences?words_or_indices=["india", 22]

$ http://localhost:8000/api/v1/get_vectors?words_or_indices=[1, "india"]

$ http://localhost:8000/api/v1/get_distance?word_or_index1=ap&word_or_index2=india

$ http://localhost:8000/api/v1/get_distances?row_words_or_indices=["india", 33]

$ http://localhost:8000/api/v1/get_nearest?v_w_i=india&k=100

$ http://localhost:8000/api/v1/get_nearest?v_w_i=india&k=100&metric=euclidean
```

> To see all API methods of wordvecspace please run http://localhost:8000/api/v1/apidoc

### Interactive console
```bash
# wordvecspace provides command to directly interact with it

$ wordvecspace interact <type> <input_dir> --metric <metric> --eargs <eargs>

# <type> is for specifying wordvecspace functionality (eg: mem, disk or annoy).
# <input_dir> is for wordvecspace data dir
# <metric> is to specify type for distance calculation
# <eargs> is for specifying extra arguments for annoy
```

Example:
```bash
# For mem
$ wordvecspace interact mem /home/user/output_dir --metric euclidean

# For Disk
$ wordvecspace interact disk /home/user/output_dir --metric euclidean

# For Annoy
$ wordvecspace interact annoy /home/user/output_dir --metric angular --eargs n_trees=1:index_fpath=/tmp
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
$ export WORDVECSPACE_DATADIR="/home/user/output_dir"

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
