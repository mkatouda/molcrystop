from setuptools import setup

setup(
    name="molcrystop",
    version="0.0.1",
    install_requires=[
        "pyyaml", 
    ],
    entry_points={
        'console_scripts': [
            'molcrystop=molcrystop.molcrystop:main',
        ],
    },
    author="Michio Katouda",
    author_email="katouda@rist.or.jp",
    description="Python script easy to make force field topology of molecular crystals",
    url="https://github.com/mkatouda/molcrystop",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.7',
)
