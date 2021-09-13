# -*- coding: utf-8 -*-

"""
 Functions for fetching synonym datasets
"""

# external imports
# ---

import os

# import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch

# internal imports
# ---

from .utils import _fetch_file


def fetch_TOEFL():
    """
    Fetch TOEFL dataset for testing synonyms

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': the questions
        'y': the answers (4 alternatives
        - the first one being the good answer)

    examples
    --------

    X : enormous-j
    y : tremendous-j    appropriate-j   unique-j    decidedly-r
    (the good answer is 'tremendous-j')

    Reference
    ---------
    Landauer, T. K., & Dumais, S. T. (1997). A solution to Plato’s problem: The Latent Semantic Analysis theory of the acquisition, induction, and representation of knowledge. Psychological Review, 104(2), 211–240. https://doi.org/10.1037/0033-295X.104.2.211

    Additional information
    ----------------------

    TOEFL Synonym Questions (State of the art)

    https://aclweb.org/aclwiki/TOEFL_Synonym_Questions_(State_of_the_art)

    Notes
    -----

    /

    """

    print("\nFetch '{}' dataset\n---\n".
          format("TOEFL"))

    # there
    url = 'file:' + os.path.expanduser('~/Downloads/TOEFL.txt')

    input_path = _fetch_file(url, 'synonymy')

    df = pd.read_csv(input_path, header=0, encoding='utf-8', sep="\t")

    data = df.values

    X = data[:, 1].astype("object")

    y = data[:, 2:].astype("object")

    return Bunch(X=X, y=y)


def fetch_ESL():
    """
    Fetch ESL dataset for testing synonyms

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': the questions
        'y': the answers (5 alternatives
        - the first one being the good answer)

    examples
    --------

    X : rusty
    y : corroded    black   dirty   painted
    (the good answer is 'corroded')

    Reference
    ---------
    Turney, P. D. (2001). Mining the Web for synonyms: PMI-IR versus LSA on TOEFL. In Proceedings of the 12th European Conference on Machine Learning (ECML-2001), (pp. 491–502). Berlin, Heidelberg: Springer. https://doi.org/10.1007/3-540-44795-4_42

    Additional information
    ----------------------

    /


    Notes
    -----

    /

    """

    print("\nFetch '{}' dataset\n---\n".
          format("ESL"))

    # there
    url = 'file:' + os.path.expanduser('~/Downloads/ESL2.txt')

    input_path = _fetch_file(url, 'synonymy')

    df = pd.read_csv(input_path, header=0, encoding='utf-8', sep="\t")

    data = df.values

    X = data[:, 1].astype("object")

    y = data[:, 2:].astype("object")

    return Bunch(X=X, y=y)
