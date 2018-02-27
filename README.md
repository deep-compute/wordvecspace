# WordVecSpace
A high performance pure python module that helps in loading and performing operations on word vector spaces created using Google's Word2vec tool.

This module has ability to load data into memory using `WordVecSpaceMem` or it can also support performing operations on the data which is on the disk using `WordVecSpaceAnnoy`.

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
# 3. Edit the command "time ./word2vec -train text8 -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15" to "time ./word2vec -train text8 -output vectors.bin -cbow 1 -size 5 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -save-vocab vocab.txt -iter 15" to get vocab.txt file also as output.
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
# <output_file> is the directory where you want to put your output file
```

### Importing

#### Quick example

##### Import
```python
>>> from wordvecspace import WordVecSpaceMem
```

##### Load data
```python
>>> wv = WordVecSpaceMem('/path/to/wvspacefile', dim)

# dim = Dimension of vectors
```

##### Make get_nearest call
```python
>>> wv.get_nearest('india', k=20)
[509, 486, 523, 4343, 14208, 13942, 42424, 25578, 6212, 2475, 3560, 13508, 20919, 3389, 4484, 19995, 8776, 7012, 12191, 16619]
```

#### Types
`wordvecspace` can perform operations by loading data into RAM using `WordVecSpaceMem` or directly on the data which is on the disk using `WordVecSpaceAnnoy`

`WordVecSpaceMem` is a bruteforce algorithm which compares given word with all the words in the vector space

`WordVecSpaceAnnoy` takes wvspace file as input and creates annoy indexes in another file. Using this file `annoy` gives approximate results quickly. For better understanding of `Annoy` please go through this [link](https://github.com/spotify/annoy)

As we have seen how to import `WordVecSpaceMem`  above, lets look at `WordVecSpaceAnnoy`

##### Import
```python
>>> from wordvecspace import WordVecSpaceAnnoy
```

##### Load data
```python
wv = WordVecSpaceAnnoy('/path/to/wvspacefile', dim, n_trees, metric='angular')

# dim = dimensions of a vector
# n_trees = number of trees(More trees gives a higher precision when querying for get_nearest)
# metric = type of distance calculation (eg: angular, euclidean)
```

##### Make get_nearest call
```python
>>> wv.get_nearest('india', k=20)
[509, 486, 523, 4343, 14208, 13942, 42424, 25578, 6212, 2475, 3560, 13508, 20919, 3389, 4484, 19995, 8776, 7012, 12191, 16619]
```

> WordVecSpaceMem and WordVecSpaceAnnoy have the same common methods.

#### Examples of using wordvecspace methods

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
  File "/usr/local/lib/python3.6/dist-packages/wordvecspace/wordvecspace_mem.py", line 196, in get_word_index
    raise UnknownWord(word)
wordvecspace.wordvecspace_mem.UnknownWord: "inidia"
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
  File "/usr/local/lib/python3.6/dist-packages/wordvecspace/wordvecspace_mem.py", line 209, in get_word_indices
    index = self.get_word_index(word, raise_exc=raise_exc)
  File "/usr/local/lib/python3.6/dist-packages/wordvecspace/wordvecspace_mem.py", line 196, in get_word_index
    raise UnknownWord(word)
wordvecspace.wordvecspace_mem.UnknownWord: "deepcompute"
```

##### Get Word at Index 
```python
# Get word at Index 509
>>> print(wv.get_word_at_index(509))
india
```
##### Get Words at Indices
```python
>>> print(wv.get_word_at_indices([1,509,71190,72000]))
['the', 'india', 'reka', None]
```

##### Get occurence of the word
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

>>> print(wv.get_word_vector("india"))
[-6.4482 -2.1636  5.7277 -3.7746  3.583 ]

# Get the unit word vector for a word india
>>> print(wv.get_word_vector("india", normalized=True))
[-0.6259 -0.21    0.5559 -0.3664  0.3478]

# Get the unit vector for a word inidia.
>>> print(wv.get_word_vector('inidia', normalized=True, raise_exc=True))
Traceback (most recent call last):
  File "/usr/lib/python3.6/code.py", line 91, in runcode
    exec(code, self.locals)
  File "<console>", line 1, in <module>
  File "/usr/local/lib/python3.6/dist-packages/wordvecspace/wordvecspace_mem.py", line 287, in get_word_vector
    index = self.get_word_index(word, raise_exc)
  File "/usr/local/lib/python3.6/dist-packages/wordvecspace/wordvecspace_mem.py", line 196, in get_word_index
    raise UnknownWord(word)
wordvecspace.wordvecspace_mem.UnknownWord: "inidia"

# Get the unit vector for a word inidia. If the word is not present it simply returns zeros if raise_exc is False.
>>> print(wv.get_word_vector('inidia', normalized=True))
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
```

##### Get distance between list of words

```python
>>> print(wv.get_distances("for", ["to", "for", "india"]))
[[  1.4990e-01]
 [ -1.1921e-07]
 [  1.3855e+00]]

>>> print(wv.get_distances("for", ["to", "for", "inidia"]))
[[  1.4990e-01]
 [ -1.1921e-07]
 [  1.0000e+00]]

>>> print(wv.get_distances(["india", "for"], ["to", "for", "usa"]))
[[  1.1830e+00   1.3855e+00   4.8380e-01]
 [  1.4990e-01  -1.1921e-07   1.4975e+00]]

>>> print(wv.get_distances(["india", "usa"]))
[[ 1.4903  0.4202  0.269  ...,  1.2041  1.3539  0.6154]
 [ 1.8084  0.9541  1.1678 ...,  0.5963  1.0458  1.1608]]

>>> print(wv.get_distances(["andhra"]))
[[ 1.3432  0.5781  0.2306 ...,  1.0937  1.1369  0.4284]]
```

##### Get nearest
```python
# Get nearest neighbours for given word or index
>>> print(wv.get_nearest("india", 20))
[509, 486, 523, 4343, 14208, 13942, 42424, 25578, 6212, 2475, 3560, 13508, 20919, 3389, 4484, 19995, 8776, 7012, 12191, 16619]

# Get nearest neighbours for given words or indices
>>> print(wv.get_nearest(["ram", "india"], 5))
[[3844, 1885, 2754, 16727, 27177], [509, 14208, 3389, 9772, 26437]]
```
### Service

```bash
# Run wordvecspace as a service (which continuously listens on some port for API requests)
$ wordvecspace runserver <type> <wvargs>

# <type> is for specifying wordvecspace functionality (eg: mem or annoy).
# <wvargs> is for specifying arguments (eg: for mem input_file=/home/user/file:dim=5:port=8000)
```

Example:

```bash
# For mem
$ wordvecspace runserver mem input_file=/home/ram/Ram/data/dc.wvspace:dim=5:port=8000

# For annoy
$ wordvecspace runserver annoy input_file=/home/ram/Ram/data/dc.wvspace:dim=5:port=8000:n_trees=1:metric=angular

# The arguments are input_file, dim, port, n_trees and metric
- input_file is for wordvecspace file
- dim is dimension of vectors in wordvecspace file
- port is to run wordvecspace in that port
- n_trees is for number of trees for annoy
- metric is type for distance calculation (eg: Euclidean, cosine)

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

$ http://localhost:8000/api/v1/get_word_vectors?words_or_indices=[1, 'india']

$ http://localhost:8000/api/v1/get_distance?word_or_index1=ap&word_or_index2=india

$ http://localhost:8000/api/v1/get_distances?row_words_or_indices=["india", 33]

$ http://localhost:8000/api/v1/get_nearest?words_or_indices=india&k=100
```

> To see all API methods of wordvecspace please run http://localhost:8000/api/v1/apidoc

### Interactive console
```bash
# For mem
$ wordvecspace interact <type> <wvargs>

# <type> is for specifying wordvecspace functionality (eg: mem or annoy).
# <wvargs> is for specifying arguments (eg: for mem input_file=/home/user/file:dim=5:port=8000)
```
Example:
```bash
# For mem
$ wordvecspace interact mem input_file=/home/user/file:dim=5:port=8000

# For annoy
$ wordvecspace interact annoy input_file=/home/user/file:dim=5:port=8000:n_trees=1:metric=angular

Total number of vectors and dimensions in wvspace file (71291, 5)

>>> help
['DEFAULT_K', '__class__', '__delattr__', '__dict__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_make_array', '_perform_sgemm', '_perform_sgemv', '_get_distances', 'get_nearest_neighbors', 'get_word_at_index', 'get_word_index', 'get_word_at_indices' 'get_word_occurrences', 'get_words_occurrences', 'get_word_vector', 'get_word_vectors', 'magnitudes', 'vectors', 'word_indices', 'word_occurrences', 'words']

WordVecSpace console
>>> wv = WordVecSpaceMem
```
## Running tests

```bash
# Download the data files
$ wget 'https://s3.amazonaws.com/deepcompute-public/data/wordvecspace/small_test_data.tgz'

# Extract downloaded small_test_data.tgz file
$ tar xvzf small_test_data.tgz

# Export the path of data files to the environment variables
$ export WORDVECSPACE_DATADIR="/home/user/small_test_data/test.wvspace"

# Run tests
$ python setup.py test
```

## GPU acceleration

`wordvecspace` can take advantage of an Nvidia GPU to perform some operations significantly faster. This is as simple as doing

```python
>>> from wordvecspace.cuda import WordVecSpace
```

The `WordVecSpace` from the `cuda` module is a drop-in replacement for the CPU based `WordVecSpace` class showcased above.

> NOTE: The vector space size must fit on available GPU ram for this to work
> Also, you will need to install cuda support by doing "sudo pip install wordvecspace[cuda]"
