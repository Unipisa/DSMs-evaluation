from gensim.models import KeyedVectors
import numpy as np
import sys
import os
import datetime
import time
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from neurora import rdm_corr
import statistics
import getopt
from utils import load_samples
from utils import load_couples_dict
from utils import create_spaces_dict
from utils import to_numpy


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

		Spaces are here evaluated on random samples composed by the same number of tokens.

		Input parameters:
		-s, the path to the sample file
		-e, the path to the file containing the list of spaces.
		-c, the path to the couple list file.
		-o, the path to the output file.  

		Output format:
		first space, second space, average of correlations, number of spurious samples

		""")


def get_embeddings(space_path, samples_list):
	#create a dictionary to save the vectors for each sample
	spaces_bucket = dict()
	#upload all the vectors of the space
	vectors_gensim = KeyedVectors.load_word2vec_format(space_path)
	for i, sample in enumerate(samples_list):
		#create a key coresponding to the sample
		spaces_bucket[i] = dict()
		vectors_sample = dict()
		#save only the words and the vectors of the sample
		vectors_sample = {word: vectors_gensim[word] for word in sample}
		spaces_bucket[i]["path"] = space_path
		spaces_bucket[i]["vectors"] = vectors_sample
		#embeddings dimension (i.e. 300, 2000, 100000, etc)
		spaces_bucket[i]["dim"] = vectors_gensim.vector_size
	return spaces_bucket



def rsa_analysis(path, couples_dict, spaces, sample_list ):
	with open(path, "w") as out:
		#upload the first space once
		for s_1 in couples_dict:
			#######log########################
			print("get embeddings of space_1: "+s_1)
			start = datetime.datetime.now() 
			#########################################
			#get the space path
			path_space_1 = spaces[s_1]["path"]
			#get embeddings takes as input the space path and the samples 
			space_list_1 = get_embeddings(path_space_1, sample_list)
			#######log########################
			print("number of samples: ",len(space_list_1))
			end = datetime.datetime.now() 
			print ("Uploading time: ", end-start)
			########################################
			#upload the other spaces
			for s_2 in couples_dict[s_1]:
				#spurious count the non significative correlations
				spurious = 0
				#if the first space and the second are equal
				if s_1 == s_2:
					correlations = []
					for i,sample_words in enumerate(sample_list):
						#convert to numpy
						numpy_arrays_1 = to_numpy(space_list_1[i]["vectors"], space_list_1[i]["dim"])
						numpy_arrays_2 = to_numpy(space_list_1[i]["vectors"], space_list_1[i]["dim"])
						#compress sparse row matrix
						A_sparse = np.matrix(numpy_arrays_1)
						B_sparse = np.matrix(numpy_arrays_2)
						#pairwise dense output
						similarities_1 = cosine_similarity(A_sparse)
						similarities_2 = cosine_similarity(B_sparse)
						#spearman correlation
						spearman = rdm_corr.rdm_correlation_spearman(similarities_1, similarities_2)
						print ("correlation sample "+str(i)+": ", s_1, s_2, i, spearman)
						# if pvalue is higher than 0.05 the correlation is not significant
						if spearman[1] > 0.05: 
							print ("Pvalue: ", spearman[1] , "spurious")
							spurious = spurious+1
						correlations.append(spearman[0])
				#if the first space and the second are not equal
				else:
					#######log########################
					print("get embeddings of space_2: "+s_2)
					start = datetime.datetime.now() 
					#########################################
					#get second space path
					path_space_2 = spaces[s_2]["path"]
					#get second space embeddings
					space_list_2 = get_embeddings(path_space_2, sample_list)
					#######log########################
					end = datetime.datetime.now() 
					print ("Uploading time: ", end-start)
					########################################
					correlations = []
					for i,sample_words in enumerate(sample_list):
						#convert to numpy
						numpy_arrays_1 = to_numpy(space_list_1[i]["vectors"], space_list_1[i]["dim"])
						numpy_arrays_2 = to_numpy(space_list_2[i]["vectors"], space_list_2[i]["dim"])
						#ompress sparse row matrix
						A_sparse = np.matrix(numpy_arrays_1)
						B_sparse = np.matrix(numpy_arrays_2)
						#pairwise dense output
						similarities_1 = cosine_similarity(A_sparse)
						similarities_2 = cosine_similarity(B_sparse)
						#spearman correlation
						spearman = rdm_corr.rdm_correlation_spearman(similarities_1, similarities_2)
						print ("correlation sample "+str(i)+": ", s_1, s_2, i, spearman)
						# if pvalue is higher than 0.05 the correlation is not significant
						if spearman[1] > 0.05: 
							print (s_1, s_2, i, spearman, "spurious")
							spurious = spurious+1
						correlations.append(spearman[0])
						numpy_arrays_1 = 0
						A_sparse = 0
						similarities_1 = 0
						numpy_arrays_2 = 0
						B_sparse = 0
						similarities_2 = 0
				avg_correlations = statistics.mean(correlations)
				outstring = "{}\t{}\t{}\t{}\n".format(s_1,s_2,str(avg_correlations),spurious)
				out.write(outstring)
	return path

#############################################

def main(argv):   
	total_start = time.time()
	try:
		opts, args = getopt.getopt(\
		argv[1:], "hs:e:c:o:",\
		["help", "samples=", "embeddings=", "couples=", "outfile="])
	except getopt.GetoptError as err:
		print (str(err))
		sys.exit(2)
	samples = None
	emb = None
	couples = None
	outfile = None
	for opt, value in opts:
		if opt in ("-h", "--help"):
			help_msg()
			sys.exit()   
		elif opt in ("-s", "--samples"):  
			samples = os.path.abspath(value)
		elif opt in ("-e", "--embeddings"):  
			emb = os.path.abspath(value)
		elif opt in ("-c", "--couples"):  
			couples = os.path.abspath(value)
		elif opt in ("-o", "--outfile"):  
			outfile = os.path.abspath(value)
	
		else:
		    assert False, "not handled option"

	if samples is None:
		assert False, "please, insert the path to samples file"
	if emb is None:
		assert False, "please, insert the path to the file containing the list of spaces"
	if couples is None:
		assert False, "please, insert the path to the couple list file"
	if outfile is None:
		assert False, "please, insert the path to the outfile"
	print("samples: ",samples)
	print("vectors: ",emb)
	print("couples: ",couples)
	print("outfile: ",outfile)
	sample_list = load_samples(samples)
	spaces_dict = create_spaces_dict(emb)
	couples_dict = load_couples_dict(couples)
	results_path = rsa_analysis(outfile, couples_dict, spaces_dict, sample_list)
      
if __name__ == '__main__':
	main(sys.argv)

