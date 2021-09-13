import numpy as np


def load_samples(path):
	sample_list = []
	with open(path, "r") as inputfile:
	    #for each sample
	    for line in inputfile:
	        # save the words in a list
	        sample_words = line.strip().split(",")
	        # save the list as set
	        sample_list.append(set(sample_words))
	return sample_list



def load_couples_dict(path):
	couples_dict = dict()
	#get spaces couples
	with open(path, "r") as couple_file:
	    for i,line in enumerate(couple_file):
	        space_1, space_2 = line.strip().split("\t")
	        if space_1 not in couples_dict:
	            couples_dict[space_1] = []
	        couples_dict[space_1].append(space_2)
	return couples_dict




def create_spaces_dict(path):
	spaces = dict()
	with open (path, "r") as spa:
	    for line in spa:
	        name, path = line.strip().split(",")
	        spaces[name] = dict()
	        spaces[name]["path"] = path
	        #get the space dimension
	        with open(path, "r") as infile:
	            for i,l in enumerate(infile):
	                line_list = l.strip().split()
	                if i ==0:
	                    if len(line_list) == 2:
	                        spaces[name]["dim"] = line_list[1]
	                    else:
	                        spaces[name]["dim"] = len(line_list)-1
	                    break
	return spaces



def to_numpy(vectors, dim):
    wvecs=np.zeros((1000,int(dim)),float)
    idx = 0
    for word in vectors:
        wvecs[idx,] = np.array(list(map(float,vectors[word])))
        idx = idx+1
    return wvecs




def sample_making(sample, vectors_gensim, space_path):
    sample_dict = dict()
    sample_dict["path"] = space_path
    sample_dict["dim"] = vectors_gensim.vector_size
    sample_dict["vectors"] = dict()
    for i,s in enumerate(sample):
        vectors_sample = dict()
        vectors_sample = {word: vectors_gensim[word] for word in s}
        sample_dict["vectors"][i] = dict()
        sample_dict["vectors"][i] = vectors_sample
    return sample_dict