from setuptools import setup, find_packages

version = '0.5.4'
setup(
    name="wordvecspace",
    python_requires='>3.5.1',
    version=version,
    description="A high performance pure python module that helps in"
                " loading and performing operations on word vector spaces"
                " created using Google's Word2vec tool. It also supports"
                " converting word vector space data (vectors and vocabulary)"
                " from Google Word2Vec format to WordVecSpace format.",
    keywords='wordvecspace',
    author='Deep Compute, LLC',
    author_email="contact@deepcompute.com",
    url="https://github.com/deep-compute/wordvecspace",
    download_url="https://github.com/deep-compute/wordvecspace/tarball/%s" % version,
    license='MIT License',
    install_requires=[
        'annoy==1.11.4',
        'scipy==1.0.0',
        'diskarray==0.1.8',
        'diskdict==0.2.2',
        'deeputil==0.2.5',
        'bottleneck==1.2.1',
    ],
    extras_require={
        'cuda': ['pycuda==2018.1.1', 'scikit-cuda==0.5.1'],
        'service': ['kwikapi[tornado]==0.4.5']
    },
    package_dir={'wordvecspace': 'wordvecspace'},
    packages=find_packages('.'),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
    ],
    test_suite='test.suite_test',
    entry_points={
        "console_scripts": [
            "wordvecspace = wordvecspace:main",
        ]
    }
)
