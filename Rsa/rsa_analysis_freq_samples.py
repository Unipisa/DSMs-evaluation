import os 
from gensim.models import KeyedVectors
import numpy as np
import sys
import getopt
import datetime
import time
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from neurora import rdm_corr
import statistics
import itertools
from utils import load_samples
from utils import load_couples_dict
from utils import create_spaces_dict
from utils import to_numpy
from utils import sample_making

def help_msg():
        print ("""
RSA is here applied to investigate the similarity of the semantic spaces of DSMs by measuring 
the correlation between the pairwise similarity relations among the lexical items in different 
spaces.

Given the large size of the DSM vocabulary (more than 345,000 words) made the construction of 
one global similarity matrix computationally too expensive, the code used takes as input a 
set of samples. 

The similarity between the semantic spaces produced by two DSMs is the average Spearman 
correlation between their respective RSMs of the various samples. 

Spaces are here evaluated on three different sets of samples, where the tokens have a high, 
a medium, or a low frequency.

Input parameters:
-l, the path to the file containing the low frequency token samples.
-m, the path to the file containing the medium frequency token samples.
-x, the path to the file containing the high frequency token samples.
-e, the path to the vector list file.
-c, the path to the couple list file.
-o, the path to the output file.  

Output format:
first space, second space, average of correlations, sample type, number of spurious samples

""")
    
def get_embeddings(space_path, sample_low, sample_medium, sample_high):
    #create a dictionary to save the vectors for each sample
    spaces_bucket_space = dict()
    #upload all the vectors of the space
    vectors_gensim = KeyedVectors.load_word2vec_format(space_path)
    print("vectors_loaded")
    #save only the words and the vectors of the sample for each sample type (i.e. low, medium, high)
    spaces_bucket_space["low"] = sample_making(sample_low, vectors_gensim, space_path)
    spaces_bucket_space["medium"] = sample_making(sample_medium, vectors_gensim,  space_path)
    spaces_bucket_space["high"] = sample_making(sample_high, vectors_gensim,  space_path)
    
    vectors_gensim = dict()
    return spaces_bucket_space



def rsa_analysis(spaces, couples_dict, samples_low, samples_medium, samples_high, correlation_reports):
    with open(correlation_reports, "w") as out:
        spaces_bucket = dict()
        #upload the first space once
        for s_1 in couples_dict:
            spaces_bucket[s_1] = dict()
            #######log########################
            print("get embeddings of space_1: "+s_1)
            start = datetime.datetime.now() 
            #########################################
            path_space_1 = spaces[s_1]["path"]
            spaces_bucket[s_1] = get_embeddings(path_space_1, samples_low, samples_medium, samples_high)
            #######log########################
            print("number of samples: ",len(spaces_bucket[s_1]))
            end = datetime.datetime.now() 
            print ("Uploading time: ", end-start)
            ########################################
            #upload the other spaces
            for s_2 in couples_dict[s_1]:
                #if the first space and the second are equal
                if s_2 == s_1:
                    # for each sample type (i.e. low, medium, high)
                    for sample, v_1 in spaces_bucket[s_1].items():
                        correlations = []
                        #spurious count the non significative correlations
                        spurious = 0
                        print ("Sample type: ",sample)
                        for i, v_2 in spaces_bucket[s_1][sample]["vectors"].items():
                            #convert to numpy
                            numpy_arrays_1 = to_numpy(spaces_bucket[s_1][sample]["vectors"][i], spaces_bucket[s_1][sample]["dim"])
                            #compress sparse row matrix
                            A_sparse = np.matrix(numpy_arrays_1)
                            #pairwise dense output
                            similarities_1 = cosine_similarity(A_sparse)
                            #spearman correlation
                            spearman = rdm_corr.rdm_correlation_spearman(similarities_1, similarities_1)
                            # if pvalue is higher than 0.05 the correlation is not significant
                            if spearman[1] > 0.05: 
                                spurious = spurious+1
                                print ("Pvalue: ", spearman[1] , "spurious")
                            correlations.append(spearman[0])
                        # average of the correlations for each sample type (i.e. low, medium, high)
                        avg_correlations = statistics.mean(correlations)
                        outstring = "{}\t{}\t{}\t{}\t{}\n".format(s_1,s_2,str(avg_correlations), sample, spurious)
                        out.write(outstring)
                        spurious = 0
                        numpy_arrays_1 = 0
                        A_sparse = 0
                        similarities_1 = 0
                 #if the first space and the second are not equal
                else:
                    spaces_bucket[s_2] = dict()
                    #######log########################
                    print("get embeddings of space_2: "+s_2)
                    start = datetime.datetime.now() 
                    #########################################
                    #get second space path
                    path_space_2 = spaces[s_2]["path"]
                    spaces_bucket[s_2]  = get_embeddings(path_space_2, samples_low, samples_medium, samples_high)
                    #######log########################
                    end = datetime.datetime.now() 
                    print ("Uploading time: ", end-start)
                    ########################################
                    # for each sample type  (i.e. low, medium, high)
                    for sample, v_1 in spaces_bucket[s_1].items():
                        print ("Sample type: ",sample)
                        correlations = []
                        spurious = 0
                        for i, v_2 in spaces_bucket[s_1][sample]["vectors"].items():
                            #convert to numpy
                            numpy_arrays_1 = to_numpy(spaces_bucket[s_1][sample]["vectors"][i], spaces_bucket[s_1][sample]["dim"])
                            numpy_arrays_2 = to_numpy(spaces_bucket[s_2][sample]["vectors"][i], spaces_bucket[s_2][sample]["dim"])
                            #compress sparse row matrix
                            A_sparse = np.matrix(numpy_arrays_1)
                            B_sparse = np.matrix(numpy_arrays_2)
                            #pairwise dense output
                            similarities_1 = cosine_similarity(A_sparse)
                            similarities_2 = cosine_similarity(B_sparse)
                            #spearman correlation
                            spearman = rdm_corr.rdm_correlation_spearman(similarities_1, similarities_2)
                            # if pvalue is higher than 0.05 the correlation is not significant
                            if spearman[1] > 0.05: 
                                spurious = spurious+1
                                print ("Pvalue: ", spearman[1] , "spurious")
                            correlations.append(spearman[0])
                        # average of the correlations for each sample type (i.e. adj, nouns, verbs)
                        avg_correlations = statistics.mean(correlations)
                        outstring = "{}\t{}\t{}\t{}\t{}\n".format(s_1,s_2,str(avg_correlations), sample, spurious)
                        out.write(outstring)
                        spurious = 0
                        numpy_arrays_1 = 0
                        A_sparse = 0
                        similarities_1 = 0
                        numpy_arrays_2 = 0
                        B_sparse = 0
                        similarities_2 = 0
                    spaces_bucket[s_2] = dict()
            spaces_bucket[s_1] = dict()     
    return correlation_reports


def main(argv):   
        total_start = time.time()
        try:
                opts, args = getopt.getopt(\
                argv[1:], "hl:m:x:e:c:o:",\
                ["help", "samples_low=", "samples_med=", "samples_high", "embeddings=", "couples=", "outfile="])
        except getopt.GetoptError as err:
                print (str(err))
                sys.exit(2)
        path_samples_low = None
        path_samples_medium = None
        path_samples_high = None
        emb = None
        couples = None
        outfile = None
        for opt, value in opts:
                if opt in ("-h", "--help"):
                        help_msg()
                        sys.exit()   
                elif opt in ("-l", "--samples_low"):  
                        path_samples_low = os.path.abspath(value)
                elif opt in ("-m", "--samples_med"):  
                        path_samples_medium = os.path.abspath(value)
                elif opt in ("-x", "--samples_high"):  
                        path_samples_high = os.path.abspath(value)
                elif opt in ("-e", "--embeddings"):  
                        emb = os.path.abspath(value)
                elif opt in ("-c", "--couples"):  
                        couples = os.path.abspath(value)
                elif opt in ("-o", "--outfile"):  
                        outfile = os.path.abspath(value)
                else:
                    assert False, "not handled option"
        if path_samples_low is None:
                assert False, "please, insert the path to low frequency sample file"
        if path_samples_medium is None:
                assert False, "please, insert the path to medium frequency sample file"
        if path_samples_high is None:
                assert False, "please, insert the path to high frequency sample file"
        if emb is None:
                assert False, "please, insert the path to the vector list file"
        if couples is None:
                assert False, "please, insert the path to the couple list file"
        if outfile is None:
                assert False, "please, insert the path to the outfile"
        print("low freq samples: ",path_samples_low)
        print("medium freq samples: ",path_samples_medium)
        print("high freq samples: ",path_samples_high)
        print("vectors: ",emb)
        print("couples: ",couples)
        print("outfile: ",outfile)
        samples_low = load_samples(path_samples_low)   
        samples_medium = load_samples(path_samples_medium) 
        samples_high = load_samples(path_samples_high) 
        spaces_dict = create_spaces_dict(emb)
        couples_dict = load_couples_dict(couples)
        results_path = rsa_analysis(spaces_dict, couples_dict, samples_low, samples_medium, samples_high, outfile)



      
if __name__ == '__main__':
    main(sys.argv)

