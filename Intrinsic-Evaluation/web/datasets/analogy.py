# -*- coding: utf-8 -*-

"""
 Functions for fetching analogy datasets
"""

# external imports
# ---

from collections import defaultdict
import glob
import os
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.datasets.base import Bunch

# internal imports
# ---
from .utils import _get_dataset_dir
from .utils import _fetch_file
from .utils import _change_list_to_np
from .utils import _get_as_pd
from ..utils import standardize_string

# TODO: rewrite to a more standarized version


def fetch_semeval_2012_2(which="all", which_scoring="golden"):
    """
    Fetch dataset used for SEMEVAL 2012 task 2 competition

    Parameters
    -------
    which : "all", "train" or "test"
    which_scoring: "golden" or "platinium" (see Notes)

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X_prot': dictionary keyed on category. Each entry is a matrix of prototype word pairs (see Notes)
        'X': dictionary keyed on category. Each entry is a matrix of question word pairs
        'y': dictionary keyed on category. Each entry is a dictionary word pair -> score

        'categories_names': dictionary keyed on category. Each entry is a human readable name of
        category.
        'categories_descriptions': dictionary keyed on category. Each entry is a human readable description of
        category.

    Reference
    ---------
    DA Jurgens et al.,
    "Measuring degrees of relational similarity. In *SEM 2012: The First Joint Conference on Lexical
    and Computational Semantics", 2012

    Notes
    -----
    Dataset used in competition was scored as in golden scoring (which_scoring) parameter, however
    organisers have released improved labels afterwards (platinium scoring)

    The task is, given two pairs of words, A:B and C:D, determine the degree to which the semantic relations between
    A and B are similar to those between C and D. Unlike the more familiar task of semantic relation identification,
    which assigns each word pair to a discrete semantic relation class, this task recognizes the continuous range of
    degrees of relational similarity. The challenge is to determine the degrees of relational similarity between a
    given reference word pair and a variety of other pairs, mostly in the same general semantic relation class as the
    reference pair.
    """

    print("\nFetch '{}' dataset\n---\n".
          format("SEMEVAL 2012 task 2 competition"))

    assert which in ['all', 'train', 'test']

    assert which_scoring in ['golden', 'platinium']

    path = _fetch_file(url="https://www.dropbox.com/sh/aarqsfnumx3d8ds/AAB05Mu2HdypP0pudGrNjooaa?dl=1",
                       data_dir="analogy",
                       uncompress=True,
                       move="EN-SEMVAL-2012-2/EN-SEMVAL-2012-2.zip",
                       verbose=0)

    train_files = set(glob.glob(os.path.join(path, "train*.txt"))) - \
        set(glob.glob(os.path.join(path, "train*_meta.txt")))

    test_files = set(glob.glob(os.path.join(path, "test*.txt"))) - \
        set(glob.glob(os.path.join(path, "test*_meta.txt")))

    if which == "train":

        files = train_files

    elif which == "test":

        files = test_files

    elif which == "all":

        files = train_files.union(test_files)

    """
    categories_names : dict from category codes to category names
    ---
    for instance:
    '3_f': 'SIMILAR_Attribute Similarity'
    '8_f': 'CAUSE-PURPOSE_Instrument:Goal'
    '9_i': 'SPACE-TIME_Attachment'
    """
    categories_names = {}

    """
    categories_descriptions : dict from category codes to category descriptions
    ---
    for instance:
    '3_f': 'X and Y both have a similar attribute or feature'
    '8_f': 'X is intended to produce Y'
    '9_i': 'an X is attached to a Y'
    """
    categories_descriptions = {}

    # Every question is formed as similarity to analogy category that is
    # posed as a list of 3 prototype word pairs

    """
    prototypes
    ---
    for instance:
    '3_f': [['rake' 'fork']
             ['valley' 'gutter']
             ['painting' 'movie']]
    '8_f': [['anesthetic' 'numbness']
             ['ballast' 'stability']
             ['camouflage' 'concealment']]
    '9_i': [['belt' 'waist']
             ['rivet' 'girder']
             ['bowler' 'head']]
    """
    prototypes = {}

    """
    questions
    ---
    for instance:
    '3_f': [['picture' 'drawing']
             ['sword' 'knife']
             ['ladder' 'stairs']
             ['shovel' 'spoon']
             ['knife' 'sword']]
    '8_f': [['joke' 'laughter']
             ['glue' 'adhesion']
             ['fire' 'warmth']
             ['education' 'enlightenment']
             ['lock' 'security']]
    '9_i': [['necklace' 'neck']
             ['bracelet' 'wrist']
             ['scarf' 'neck']
             ['ring' 'finger']
             ['shoe' 'foot']]
    """
    questions = defaultdict(list)

    """
    scores : dict of dict from category codes to a list of scores
    ---
    '3_f': [41.7, 47.9, 41.7, 28.6, 28.0]
    '8_f': [62.0, 52.7, 50.0, 44.0, 40.0]
    '9_i': [56.0, 48.0, 46.0, 38.0, 38.0]

    NOTE: the scores are either of type 'golden' or 'platinium'
    In the example above they are of the type 'golden'
    ---
    """

    golden_scores = {}

    platinium_scores = {}

    scores = {"golden": golden_scores, "platinium": platinium_scores}

    for f in files:

        with open(f[0:-4] + "_meta.txt") as meta_f:

            meta = meta_f.read().splitlines()[1].split(",")

        with open(os.path.dirname(f) + "/pl-" + os.path.basename(f)) as f_pl:

            platinium = f_pl.read().splitlines()

        with open(f) as f_gl:

            golden = f_gl.read().splitlines()

        assert platinium[0] == golden[0], ("Incorrect file for ", f)

        c = meta[0] + "_" + meta[1]

        categories_names[c] = meta[2] + "_" + meta[3]

        categories_descriptions[c] = meta[4]

        prototypes[c] = [l.split(":") for l in
                         platinium[0].replace(": ", ":").replace(" ", ",").replace(".", "").split(",")]

        golden_scores[c] = {}

        platinium_scores[c] = {}

        questions_raw = []

        for line_pl in platinium[1:]:

            word_pair, score = line_pl.split()

            questions_raw.append(word_pair)

            questions[c].append([standardize_string(w) for w in word_pair.split(":")])

            # platinium_scores[c][word_pair] = score

            platinium_scores[c][word_pair] = float(score)

            # Note: not converting the string score to a float
            # will falsify the Spearman correlation

        for line_g in golden[1:]:

            word_pair, score = line_g.split()

            # golden_scores[c][word_pair] = score

            golden_scores[c][word_pair] = float(score)

        # Make scores a list

        platinium_scores[c] = [platinium_scores[c][w] for w in questions_raw]

        golden_scores[c] = [golden_scores[c][w] for w in questions_raw]

    return Bunch(X_prot=_change_list_to_np(prototypes),
                 X=_change_list_to_np(questions),
                 y=scores[which_scoring],
                 categories_names=categories_names,
                 categories_descriptions=categories_descriptions)


def fetch_msr_analogy():
    """
    Fetch MSR dataset for testing performance on syntactic analogies

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of word questions
        'y': vector of answers
        'category': name of category
        'category_high_level': name of high level category (noun/adjective/verb)

    Reference
    ---------
    Originally published at http://research.microsoft.com/en-us/projects/rnn/.

    Notes
    -----
    Authors description: "more precisely, we tagged 267M words of newspaper text
    with Treebank POS tags (Marcus et al., 1993). We then selected 100 of the most frequent comparative adjectives
    (words labeled JJR); 100 of the most frequent plural nouns (NNS); 100 of the most frequent possessive nouns
    (NN POS); and 100 of the most frequent base form verbs (VB).
    We then systematically generated analogy questions by randomly matching each of the 100 words with 5 other words
    from the same category, and creating variants.
    """

    print("\nFetch '{}' dataset\n---\n".
          format("MSR analogy"))

    url = "https://www.dropbox.com/s/ne0fib302jqbatw/EN-MSR.txt?dl=1"

    with open(_fetch_file(url, "analogy/EN-MSR", verbose=0), "r") as f:

        L = f.read().splitlines()

    """
    Typical 4 words analogy questions
    ---
    first five data points as an example:

    good better rough JJ_JJR rougher
    better good rougher JJR_JJ rough
    good best rough JJ_JJS roughest
    best good roughest JJS_JJ rough
    best better roughest JJS_JJR rougher

    """

    """
    questions:
    ---
    first five data points as an example:

        good better rough
        better good rougher
        good best rough
        best good roughest
        best better roughest
    """

    questions = []

    """
    answers:
    ---
    first five data points as an example:

        rougher
        rough
        roughest
        rough
        rougher

    """

    answers = []

    """
    category:
    ---
    first five data points as an example:

        JJ_JJR
        JJR_JJ
        JJ_JJS
        JJS_JJ
        JJS_JJR

    """

    category = []

    for l in L:

        words = standardize_string(l).split()

        questions.append(words[0:3])

        answers.append(words[4])

        category.append(words[3])

    verb = set([c for c in set(category) if c.startswith("VB")])

    noun = set([c for c in set(category) if c.startswith("NN")])

    """
    category_high_level:
    ---
    first five data points as an example:

        adjective
        adjective
        adjective
        adjective
        adjective

    """

    category_high_level = []

    for cat in category:

        if cat in verb:

            category_high_level.append("verb")

        elif cat in noun:

            category_high_level.append("noun")

        else:

            category_high_level.append("adjective")

    assert set([c.upper() for c in category]) == set(['VBD_VBZ', 'VB_VBD', 'VBZ_VBD',
                                                      'VBZ_VB', 'NNPOS_NN', 'JJR_JJS', 'JJS_JJR', 'NNS_NN', 'JJR_JJ',
                                                      'NN_NNS', 'VB_VBZ', 'VBD_VB', 'JJS_JJ', 'NN_NNPOS', 'JJ_JJS', 'JJ_JJR'])

    return Bunch(X=np.vstack(questions).astype("object"),
                 y=np.hstack(answers).astype("object"),
                 category=np.hstack(category).astype("object"),
                 category_high_level=np.hstack(category_high_level).astype("object"))


def fetch_google_analogy():
    """
    Fetch Google dataset for testing both semantic and syntactic analogies.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of word questions
        'y': vector of answers
        'category': name of category
        'category_high_level': name of high level category (semantic/syntactic)

    Reference
    ---------
    Mikolov, Tomas and Sutskever, Ilya and Chen, Kai and Corrado, Greg S and Dean, Jeff,
    "Distributed representations of words and phrases and their compositionality", 2013

    Notes
    -----
    This dataset is a subset of WordRep dataset.

    """

    print("\nFetch '{}' dataset\n---\n".
          format("Google analogy"))

    url = "https://www.dropbox.com/s/eujtyfb5zem1mim/EN-GOOGLE.txt?dl=1"

    with open(_fetch_file(url, "analogy/EN-GOOGLE", verbose=0), "r") as f:

        L = f.read().splitlines()

    # Simple 4 word analogy questions with categories

    questions = []

    answers = []

    category = []

    cat = None

    for l in L:

        if l.startswith(":"):

            cat = l.lower().split()[1]

        else:

            words = standardize_string(l).split()

            questions.append(words[0:3])

            answers.append(words[3])

            category.append(cat)

    assert set(category) == set(['gram3-comparative', 'gram8-plural', 'capital-common-countries',
                                 'city-in-state', 'family', 'gram9-plural-verbs', 'gram2-opposite',
                                 'currency', 'gram4-superlative', 'gram6-nationality-adjective',
                                 'gram7-past-tense',
                                 'gram5-present-participle', 'capital-world', 'gram1-adjective-to-adverb'])

    syntactic = set([c for c in set(category) if c.startswith("gram")])

    category_high_level = []

    for cat in category:

        category_high_level.append("syntactic" if cat in syntactic else "semantic")

    # dtype=object for memory efficiency

    return Bunch(X=np.vstack(questions).astype("object"),
                 y=np.hstack(answers).astype("object"),
                 category=np.hstack(category).astype("object"),
                 category_high_level=np.hstack(category_high_level).astype("object"))


def fetch_wordrep(subsample=None, rng=None):
    """
    Fetch MSR WordRep dataset for testing both syntactic and semantic dataset

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of word pairs
        'y': vector of answers
        'category': name of category
        'category_high_level': name of high level category (semantic/syntactic)

    Reference
    ---------
    Gao, Bin and Bian, Jiang and Liu, Tie-Yan,
    "Wordrep: A benchmark for research on learning word representations", 2014


    Notes
    -----
    This dataset is too big to calculate and store all word analogy quadruples, this is
    why it returns word pairs

    """

    print("\nFetch '{}' dataset\n---\n".
          format("wordrep"))

    path = _fetch_file(url="https://www.dropbox.com/sh/5k78h9gllvc44vt/AAALLQq-Bge605OIMlmGBbNJa?dl=1",
                       data_dir="analogy",
                       uncompress=True,
                       move="EN-WORDREP/EN-WORDREP.zip",
                       verbose=0)

    wikipedia_dict = glob.glob(os.path.join(path, "Pairs_from_Wikipedia_and_Dictionary/*.txt"))

    wordnet = glob.glob(os.path.join(path, "Pairs_from_WordNet/*.txt"))

    # This dataset is too big to calculate and store all word analogy quadruples

    """
    word pairs
    ---
    first five data points as an example:

    internal    internally
    grateful    gratefully
    bright  brightly
    helpless    helplessly
    oral    orally
    """
    word_pairs = []

    """
    categories
    ---
    first five data points as an example:

    adjective-to-adverb
    adjective-to-adverb
    adjective-to-adverb
    adjective-to-adverb
    adjective-to-adverb
    """

    category = []

    """
    high level categories
    ---
    first five data points as an example:

    wikipedia-dict
    wikipedia-dict
    wikipedia-dict
    wikipedia-dict
    wikipedia-dict
    """

    category_high_level = []

    files = wikipedia_dict + wordnet

    for file_name in files:

        c = os.path.basename(file_name).split(".")[0]

        c = c[c.index("-") + 1:]

        with open(file_name, "r") as f:

            for l in f.read().splitlines():

                word_pairs.append(standardize_string(l).split())

                category.append(c)

                category_high_level.append("wikipedia-dict" if file_name in wikipedia_dict else "wordnet")

    if subsample:

        assert 0 <= subsample <= 1.0

        rng = check_random_state(rng)

        ids = rng.choice(range(len(word_pairs)), int(subsample * len(word_pairs)), replace=False)

        word_pairs = [word_pairs[i] for i in ids]

        category = [category[i] for i in ids]

        category_high_level = [category_high_level[i] for i in ids]

    wordnet_categories = {'Antonym',
                          'Attribute',
                          'Causes',
                          'DerivedFrom',
                          'Entails',
                          'HasContext',
                          'InstanceOf',
                          'IsA',
                          'MadeOf',
                          'MemberOf',
                          'PartOf',
                          'RelatedTo',
                          'SimilarTo'}

    wikipedia_categories = {'adjective-to-adverb',
                            'all-capital-cities',
                            'city-in-state',
                            'comparative',
                            'currency',
                            'man-woman',
                            'nationality-adjective',
                            'past-tense',
                            'plural-nouns',
                            'plural-verbs',
                            'present-participle',
                            'superlative'}

    return Bunch(category_high_level=np.array(category_high_level),
                 X=np.array(word_pairs),
                 category=np.array(category),
                 wikipedia_categories=wordnet_categories,
                 wordnet_categories=wikipedia_categories)


def fetch_SAT():
    """
    Fetch SAT dataset for testing analogies

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': the questions
        'y': the answers (between 4 and 5 alternatives
        - the first one is the good answer)

    examples
    --------

    X : lull-v_trust-n
    y : ['cajole-v_compliance-n' 'balk-v_fortitude-n' 'betray-v_loyalty-n'
        'hinder-v_destination-n' 'soothe-v_passion-n']


    Reference
    ---------
    Turney, P. D., Littman, M. L., Bigham, J., & Shnayder, V. (2003). Combining independent modules to solve multiple-choice synonym and analogy problems. In Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP-03).

    Additional information
    ----------------------

    SAT Analogy Questions (State of the art)

    https://aclweb.org/aclwiki/SAT_Analogy_Questions_(State_of_the_art)

    Notes
    -----

    /

    """

    print("\nFetch '{}' dataset\n---\n".
          format("SAT"))

    url = 'file:' + os.path.expanduser('~/Downloads/sat.txt')

    input_path = _fetch_file(url, 'analogy')

    df = pd.read_csv(input_path, header=0, encoding='utf-8', sep="\t")

    data = df.values

    X = data[:, 1].astype("object")

    y = data[:, 2:].astype("object")

    return Bunch(X=X, y=y)


def fetch_BATS():
    """
    Fetch BATS dataset for testing analogies

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:

    examples
    --------

    X : [['angry' 'angrier']
        ['cheap' 'cheaper']
        ['clever' 'cleverer']
        ['coarse' 'coarser']
        ['costly' 'costlier']]

    category:

    {'adj - comparative' 'adj - comparative' 'adj - comparative'
     'adj - comparative' 'adj - comparative'}

    category_high_level:

    ['Inflectional_morphology' 'Inflectional_morphology'
     'Inflectional_morphology' 'Inflectional_morphology'
     'Inflectional_morphology']

    WARNING:

    some columns in X contain multiple words separated by a /

    Here is an example from 'antonyms - gradable'

    able    unable/incapable/incompetent/unequal
    abundant    scarce/rare/tight/meager/meagre/meagerly/stingy/scrimpy/insufficient/deficient


    Reference
    ---------
    Gladkova, A., Drozd, A., & Matsuoka, S. (2016). Analogy-Based Detection of Morphological and Semantic Relations with Word Embeddings: What Works and What Doesn’t. In Proceedings of the NAACL-HLT SRW (pp. 47–54). San Diego, California, June 12-17, 2016: ACL. https://doi.org/10.18653/v1/N16-2002

    Additional information
    ----------------------

    The Bigger Analogy Test Set (BATS)
    http://vecto.space/projects/BATS/

    Notes
    -----

    /

    """

    print("\nFetch '{}' dataset\n---\n".
          format("BATS"))

    url = 'file:' + os.path.expanduser('~/Downloads/BATS_3.0.zip')

    input_folder = _fetch_file(url=url,
                               data_dir="analogy",
                               uncompress=True,
                               verbose=1)

    input_folder = os.path.join(input_folder, "BATS_3.0")

    high_level_categories = ['Inflectional_morphology',
                             'Derivational_morphology',
                             'Encyclopedic_semantics',
                             'Lexicographic_semantics']

    categories_high_level = []
    categories = []
    # questions = []
    # answers = []
    words_pairs = []

    for i, category_high_level in enumerate(high_level_categories, 1):

        sub_folder = str(i) + "_" + category_high_level

        folder = os.path.join(input_folder, sub_folder)

        for filename in os.listdir(folder):

            category = filename[5:-5].lower()

            input_file = os.path.join(folder, filename)

            with open(input_file, 'r') as file:

                for line in file:

                    line = line.strip()

                    if line:

                        # question, answer = line.split("\t")

                        # questions.append(question)

                        # answers.append(answer)

                        words_pair = line.lower().split("\t")

                        words_pairs.append(words_pair)

                        categories.append(category)

                        categories_high_level.append(category_high_level)

    # b = Bunch(X=np.hstack(questions).astype("object"),

    #              y=np.hstack(answers).astype("object"),

    #              categories=np.hstack(categories).astype("object"),

    #              categories_high_level=np.hstack(categories_high_level).astype("object"))

    b = Bunch(X=np.array(words_pairs).astype("object"),

              category=np.hstack(categories).astype("object"),

              category_high_level=np.hstack(categories_high_level).astype("object"))

    return b
