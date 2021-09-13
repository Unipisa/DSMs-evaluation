# -*- coding: utf-8 -*-

"""
 Functions to count the number of items
"""

# external imports
# ---
import numpy as np

import importlib
import os

# internal imports
# ---

# from . import similarity
from web.datasets import analogy
# from . import categorization

# from utils import number_permutations
from web.datasets.items.utils import number_permutations


def count_Xy_items(module_name, dataset_name, **kwargs):
    """
        Count the number of items in a dataset with X and y variables
        by specifying its module name (similarity, synonymy, categorization...)
        and display a sample of the data for checking purposes
    """

    # set the fetch function name
    # ---
    fetch_function_name = "fetch_" + dataset_name

    # load module
    # ---

    module = importlib.import_module("web.datasets." + module_name)

    # retrieve the dataset
    # ---
    data = getattr(module, fetch_function_name)(**kwargs)

    X = data.X

    y = data.y

    # display a short sample of the data
    # ---

    limit = 5

    for i in range(limit):

        print(i + 1, X[i], y[i])

    print("---")

    # items counting
    # ---

    n = data.X.shape[0]

    print("number of items = ", n)

    return(n)


def count_semeval_2012_2(which="all"):
    """
        Return the number of items in semeval_2012_2
        and display a sample of the data
        for checking purposes
    """

    data = analogy.fetch_semeval_2012_2(which)

    X_prot = data.X_prot

    X = data.X

    y = data.y

    categories_names = data.categories_names

    categories_descriptions = data.categories_descriptions

    # display a sample
    # ---
    categories = ('3_f', '8_f', '9_i')

    limit = 5

    for category in categories:

        print("")
        print(category)
        print("---")
        print("")
        print(categories_names[category])
        print(categories_descriptions[category])
        print(X_prot[category])
        print(X[category][:limit, :])
        print(y[category][:limit])

    # items counting
    # ---

    n = 0

    for category in categories_names:

        nb_questions = X[category].shape[0]

        n += nb_questions

    print("---")

    print("number of items = ", n)

    return(n)


def count_mikolov(corpus_name):
    """
        Return the number of items in msr_analogy or google_analogy
        and display a sample of the data
        for checking purposes
    """

    # set the fetch function name
    # ---
    fetch_function_name = "fetch_" + corpus_name

    # retrieve the dataset
    # ---
    data = getattr(analogy, fetch_function_name)()

    X = data.X

    y = data.y

    categories = data.category

    categories_high_level = data.category_high_level

    # display a sample
    # ---

    limit = 5

    print(X[:limit, :limit])
    print(y[:limit])
    print(categories[:limit])
    print(categories_high_level[:limit])

    print("---")

    # items counting
    # ---

    n = y.shape[0]

    print("number of items = ", n)

    return(n)


def count_wordrep(subsample=None, rng=None):
    """
        Return the number of items in wordrep
        and display a sample of the data
        for checking purposes
    """

    data = analogy.fetch_wordrep(subsample, rng)

    X = data.X

    categories = data.category

    categories_high_level = data.category_high_level

    wordnet_categories = data.wordnet_categories

    wikipedia_categories = data.wikipedia_categories

    # display a sample
    # ---

    limit = 5

    print(X[:limit])
    print(categories[:limit])
    print(categories_high_level[:limit])

    print("---")
    print("")
    print("WordNet categories:")
    print("---")
    print(wordnet_categories)
    print("")
    print("Wikipedia categories:")
    print("---")
    print(wikipedia_categories)

    # items counting
    # ---

    total_nb_items = 0

    p = X.shape[0]

    total_nb_pairs = 0

    print("")
    print("Statistics")
    print("---")
    print("")

    for category in wordnet_categories | wikipedia_categories:

        pairs = X[categories == category]

        nb_pairs = len(pairs)

        nb_permu = number_permutations(2, nb_pairs)

        print(category, " : ", nb_pairs, " pairs, ", nb_permu, " permutations ")

        total_nb_items += nb_permu

        total_nb_pairs += nb_pairs

    print("---")

    print("number of words pairs = ", total_nb_pairs)

    print("number of items (i.e., number of permutations) = ", total_nb_items)

    assert p == total_nb_pairs, "problem: numbers should be identical, p = " + str(p) + " total_nb_pairs = " + str(total_nb_pairs)

    return(total_nb_items)


def count_BATS():
    """
        Return the number of items in BATS
        and display a sample of the data
        for checking purposes
    """

    import numpy as np

    data = analogy.fetch_BATS()

    X = data.X

    categories = data.category

    categories_high_level = data.category_high_level

    # display a sample
    # ---

    limit = 5

    print(X[:limit])
    print(categories[:limit])
    print(categories_high_level[:limit])

    # items counting
    # ---

    total_nb_items = 0

    p = X.shape[0]

    total_nb_pairs = 0

    print("")
    print("Statistics")
    print("---")
    print("")

    for category in np.unique(categories):

        pairs = X[categories == category]

        nb_pairs = len(pairs)

        nb_permu = number_permutations(2, nb_pairs)

        print(category, " : ", nb_pairs, " pairs, ", nb_permu, " permutations ")

        total_nb_items += nb_permu

        total_nb_pairs += nb_pairs

    print("---")

    print("number of words pairs = ", total_nb_pairs)

    print("number of items (i.e., number of permutations) = ", total_nb_items)

    assert total_nb_pairs == p, "problem: numbers should be identical, p = " + str(p) + " total_nb_pairs = " + str(total_nb_pairs)

    return(total_nb_items)


def count_synonymy():
    """

    """
    assert count_Xy_items("synonymy", "TOEFL") == 80

    assert count_Xy_items("synonymy", "ESL") == 50


def count_similarity():
    """

    """

    assert count_Xy_items("similarity", "RG65") == 65

    assert count_Xy_items("similarity", "RW") == 2034

    assert count_Xy_items("similarity", "SimLex999") == 999

    assert count_Xy_items("similarity", "multilingual_SimLex999", which="EN") == 999
    assert count_Xy_items("similarity", "multilingual_SimLex999", which="DE") == 999
    assert count_Xy_items("similarity", "multilingual_SimLex999", which="IT") == 999
    assert count_Xy_items("similarity", "multilingual_SimLex999", which="RU") == 999

    assert count_Xy_items("similarity", "SimVerb3500") == 3500

    assert count_Xy_items("similarity", "WS353", which="all") == 353
    assert count_Xy_items("similarity", "WS353", which="similarity") == 203
    assert count_Xy_items("similarity", "WS353", which="relatedness") == 252

    assert count_Xy_items("similarity", "MTurk") == 287

    assert count_Xy_items("similarity", "MEN", which="all") == 3000

    assert count_Xy_items("similarity", "TR9856") == 9856


def count_analogy():
    """

    """

    assert count_Xy_items("analogy", "SAT") == 374

    assert count_mikolov("msr_analogy") == 8000

    assert count_mikolov("google_analogy") == 19544

    assert count_wordrep() == 237409102

    assert count_BATS() == 98000

    assert count_semeval_2012_2("all") == 3218


def count_categorization():
    """

    """

    assert count_Xy_items("categorization", "BLESS") == 200

    assert count_Xy_items("categorization", "AP") == 402

    assert count_Xy_items("categorization", "battig") == 5231
    assert count_Xy_items("categorization", "battig2010") == 82

    assert count_Xy_items("categorization", "ESSLLI_1a") == 44
    assert count_Xy_items("categorization", "ESSLLI_2b") == 40
    assert count_Xy_items("categorization", "ESSLLI_2c") == 45


def count_all():
    """

    """

    print("Count the items...")

    # count_Xy_items("synonymy", "TOEFL")
    # count_Xy_items("synonymy", "ESL")

    # count_Xy_items("similarity", "RG65")

    # count_Xy_items("similarity", "RW")

    # count_Xy_items("similarity", "SimLex999")
    # count_Xy_items("similarity", "multilingual_SimLex999", which="EN")
    # count_Xy_items("similarity", "multilingual_SimLex999", which="DE")
    # count_Xy_items("similarity", "multilingual_SimLex999", which="IT")
    # count_Xy_items("similarity", "multilingual_SimLex999", which="RU")
    # count_Xy_items("similarity", "SimVerb3500")
    # count_Xy_items("similarity", "WS353", which="all")
    # count_Xy_items("similarity", "WS353", which="similarity")
    # count_Xy_items("similarity", "WS353", which="relatedness")
    # count_Xy_items("similarity", "MTurk")
    # count_Xy_items("similarity", "MEN", which="all")

    # count_mikolov("msr_analogy")
    # count_mikolov("google_analogy")
    # count_wordrep()
    count_BATS()
    count_semeval_2012_2("all")
    count_Xy_items("analogy", "SAT")

    # count_Xy_items("categorization", "BLESS")
    # count_Xy_items("categorization", "AP")
    # count_Xy_items("categorization", "battig")
    # count_Xy_items("categorization", "battig2010")
    # count_Xy_items("categorization", "ESSLLI_1a")
    # count_Xy_items("categorization", "ESSLLI_2b")
    # count_Xy_items("categorization", "ESSLLI_2c")


if __name__ == "__main__":

    # count_synonymy()
    # count_similarity()
    # count_analogy()
    # count_categorization()

    count_all()

    print("--- THE END ---")
