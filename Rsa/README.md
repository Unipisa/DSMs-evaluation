## Representational Similarity Analysis (RSA)

RSA is here applied to investigate the similarity of the semantic spaces of DSMs by measuring the correlation between the pairwise similarity relations among the lexical items in different spaces.

Given the large size of the DSM vocabulary (more than 345,000 words) made the construction of one global similarity matrix computationally too expensive, the code used takes as input a set of samples. 

The similarity between the semantic spaces produced by two DSMs is the average Spearman correlation between their respective RSMs of the various samples. 

We provide code for three different kinds of evaluation, according to the examined samples:

### Random
Spaces are here evaluated on random samples composed by the same number of tokens.
Launch from command line:
```
python ./rsa_analysis_random_samples.py -s <path_to_sample_file> -e <path_to_space_list_file> -c <path_to_couple_list_file> -o <path_to_output_file>
```


### Frequency
Spaces are evaluated on three different sets of samples, where the tokens have a high, a medium, or a low frequency.
Launch from command line:
```
python ./rsa_analysis_freq_samples.py -l <path_to_low_freq_sample_file> -m <path_to_low_freq_sample_file> -x <path_to_high_freq_sample_file> -e <path_to_space_list_file> -c <path_to_couple_list_file> -o <path_to_output_file> 
```
      
### POS

Spaces are evaluated on three different sets of samples, where the tokens are adjectives, nouns, or verbs.
Launch from command line:
```
python ./rsa_analysis_pos_samples.py -a <path_to_adj_sample_file> -n <path_to_noun_sample_file> -v <path_to_verb_sample_file> -e <path_to_space_list_file> -c <path_to_couple_list_file> -o <path_to_output_file>
```


### Input and output description
#### Sample File

For a given file of samples, each row contains the tokens regarding a single sample. Token must be written comma separated.
```
Row 1: <token1>,<token2>...
Row 2: <token1>,<token2>...
...
```
#### Space list file

This file contains the list of DSMs.
```
Row 1: <model name>,<model path>
Row 2: <model name>,<model path>
```

#### Couple list file

This file contains the list of the couples of DSMs to analyse. Models' names are separated by tabulation.
```
Row 1: <model name>\t<model name>
Row 2: <model name>\t<model name>
```


#### Output file

This file contains the analysis results.
```
first space, second space, average of correlations, number of spurious samples
```
For what concerns the Frequency and POS sample analysis, the output contains also the type of sample (i.e. "low", "medium" or "high" for frequency samples; "adj", "noun" or "verb" for POS samples)

```
first space, second space, average of correlations, sample type, number of spurious samples
```
