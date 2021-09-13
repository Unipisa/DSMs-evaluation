import os
from math import factorial


def number_permutations(k, n):
    """
    Calculate the number of permutations of k elements
    chosen in a set of n elements
    """

    # print("permutation of k", k, "n", n)

    if k > n:
        raise ValueError("k cannot be greater than n, but k = {} > n = {}. ".format(k, n))

    return int(factorial(n) / factorial(n - k))


def attend_output_folder(output_folder):
    '''

    '''

    # Create the output folder recursively, if it does not exist
    # ---
    if not os.path.exists(output_folder):

        try:

            os.makedirs(output_folder)

        except Exception:
            # It has likely been created in the meantime by another process.
            # It therefore generates an exception.
            # Not a big deal.
            pass
