# # # # # # # # # # # # # # # # # # # #
# Classifying the Digital Deciders:
# The Choice Between an Open Internet and a Sovereign One
# Fahmida Y Rashid (fr48)
# Part 2: Comparing the corpus, cybersecurity
# # # # # # # # # #

'''
On the international stage, there are two visions of the Internet: an Internet open and free for all ideas and an
Internet that should be restricted to within their boundaries to show only "approved" ideas. The United States and
many other Western countries favor an open Internet, while Russia, China, and other restrictive societies prefer
the "sovereign" Internet.

New America (https://www.newamerica.org/cybersecurity-initiative/reports/digital-deciders/analyzing-the-clusters)
defines Digital Deciders as countries that have not yet picked a side.

We apply natural language processing tools on General Debate speeches made in the United Nations General Assembly
from 1970 to 2018 to determine which worldview the Digital Deciders are more likely to lean towards.

[Story on GitHub](https://fr48.github.io/prtfl)
[Project on GitHub](https://github.com/fr48/unga)
'''

'''
## Part Two: Compare the corpus

Now that I have figured out what speeches go in which corpus, I create the corpus and then compare them. 

* Read in the relevant speeches (they have already been cleaned up)
* Run TF/IDF on the corpus
* Run TF/IDF on the "unknown" corpus
* Similarity Analysis

Just in case, I am exporting the analyzed dataframes into csv files into the `output` directory.
'''

###################
# Import packages #
###################
# nltk imports
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

# processing data
import pandas as pd
import re

# feature/text extractions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# similarity
from sklearn.metrics.pairwise import cosine_similarity

##################
# Define Groups
##################

digital = ['ALB', 'ARG', 'ARM', 'BOL', 'BIH', 'BWA', 'BRA', 'COL', 'COG', 'CRI',
           'CIV', 'DOM', 'ECU', 'SLV', 'GEO', 'GHA', 'GTM', 'HND', 'IND', 'IDN',
           'IRQ', 'JAM', 'JOR', 'KEN', 'KWT', 'KGZ', 'LBN', 'MKD', 'MYS', 'MEX',
           'MNG', 'MAR', 'NAM', 'NIC', 'NGA', 'PAK', 'PAN', 'PNG', 'PRY', 'PER',
           'PHL', 'MDA', 'SRB', 'SGP', 'ZAF', 'LKA', 'THA', 'TUN', 'UKR', 'URY']

global_open = ['GBR', 'CAN', 'AUS', 'DEU', 'JPN', 'SWE', 'NLD', 'USA', 'NOR',
               'FIN', 'CHE', 'EST', 'ESP', 'POL', 'NZL', 'KOR', 'AUT', 'IRL', 'CZE',
               'PRT', 'DNK', 'ITA', 'LVA', 'LTU', 'LUX', 'BEL', 'SVN', 'GRC', 'CHL',
               'CYP', 'SVK', 'ISR', 'HRV', 'BGR', 'HUN', 'ROU', 'FRA']

sovereign_controlled = ['SAU', 'ZWE', 'VEN', 'SWZ', 'CUB', 'IRN', 'DZE', 'LBY',
                        'QAT', 'TUR', 'ARE', 'BLR', 'RUS', 'CHN', 'KAZ', 'OMN', 'BHR', 'AZE', 'VNM',
                        'CMR', 'TKM', 'TJK', 'SYR', 'PRK', 'UZB', 'AGO', 'EGY']

#################
# Cybersecurity
#################

file_list = []
file_name = "un/output/cs_deciders.csv"
with open(file_name) as f_input:
    file_list = f_input.read().split('\n')

## Run the Deciders Against the Corpus to get comparison
deciders = []
for one_file in file_list:
    decider = "un/data/"+one_file
    with open(decider) as f_input:
        deciders.append(f_input.read())

sovereigns = []
file_name = "un/output/cs_sov_corpus.csv"
with open(file_name) as f_input:
    sovereigns = f_input.read().split('\n')

currents = []
file_name = "un/output/cs_op_corpus.csv"
with open(file_name) as f_input:
    currents = f_input.read().split('\n')

## Open Internet TF/IDF

#define the vectorizer, and the ngram range (one to three words)
o_vectorizer = TfidfVectorizer(ngram_range=(1, 3))

#fit the corpus to the vectorizer, and get the feature names
o_vectors = o_vectorizer.fit_transform(currents)
o_feature_names = o_vectorizer.get_feature_names()
o_dense = o_vectors.todense()
o_denselist = o_dense.tolist()

#put the resulting calculations into a df and use feature_names for column names
oc_df = pd.DataFrame(o_denselist, columns=o_feature_names)

#use the same open vectorizers i used for the deciders
covectors = o_vectorizer.transform(deciders)
cofeature_names = o_vectorizer.get_feature_names()
codense = covectors.todense()
codenselist = codense.tolist()

#deciders using the open vectorizer
co_df = pd.DataFrame(codenselist, columns=cofeature_names, index=file_list)

# there is a lot of 0.00 so just pull out a few specific features

# compare deciders with open
counts = cosine_similarity(oc_df,co_df)
cs_open = pd.DataFrame(counts.T, index=file_list)
cs_open=cs_open.rename(columns={0: "open"})

## Sovereign Internet TF/IDF

#define the vectorizer, and the ngram range (one to three words)
s_vectorizer = TfidfVectorizer(ngram_range=(1, 3))

#fit the corpus to the vectorizer, and get the feature names
s_vectors = s_vectorizer.fit_transform(sovereigns)
s_feature_names = s_vectorizer.get_feature_names()
s_dense = s_vectors.todense()
s_denselist = s_dense.tolist()

#put the resulting calculations into a df and use feature_names for column names
sc_df = pd.DataFrame(s_denselist, columns=s_feature_names)

#use the same sovereign vectorizers i used
csvectors = s_vectorizer.transform(deciders)
csfeature_names = s_vectorizer.get_feature_names()
csdense = csvectors.todense()
csdenselist = csdense.tolist()

#deciders using the sovereign vectorizer
cs_df = pd.DataFrame(csdenselist, columns=csfeature_names, index=file_list)

# compare deciders with sovereign
counts = cosine_similarity(sc_df,cs_df)
cs_sovereign = pd.DataFrame(counts.T, index=file_list)
cs_sovereign=cs_sovereign.rename(columns={0: "sovereign"})

# compare in one table
cs = pd.concat([cs_open, cs_sovereign], axis=1, sort=False)

#################
# Economics
#################

file_list = []
file_name = "un/output/ec_deciders.csv"
with open(file_name) as f_input:
    file_list = f_input.read().split('\n')

## Run the Deciders Against the Corpus to get comparison
deciders = []
for one_file in file_list:
    decider = "un/data/"+one_file
    with open(decider) as f_input:
        deciders.append(f_input.read())

sovereigns = []
file_name = "un/output/ec_sov_corpus.csv"
with open(file_name) as f_input:
    sovereigns = f_input.read().split('\n')

currents = []
file_name = "un/output/ec_op_corpus.csv"
with open(file_name) as f_input:
    currents = f_input.read().split('\n')

## Open Internet TF/IDF

#define the vectorizer, and the ngram range (one to two words)
#3-gram would be nice but I keep running out of memory
eo_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

#fit the corpus to the vectorizer, and get the feature names
eo_vectors = eo_vectorizer.fit_transform(currents)
eo_feature_names = eo_vectorizer.get_feature_names()
eo_dense = eo_vectors.todense()
eo_denselist = eo_dense.tolist()

#put the resulting calculations into a df and use feature_names for column names
eo_df = pd.DataFrame(eo_denselist, columns=eo_feature_names)

#use the same open vectorizers i used for the deciders
eo_vectors = eo_vectorizer.transform(deciders)
eo_feature_names = eo_vectorizer.get_feature_names()
eo_dense = eo_vectors.todense()
eo_denselist = eo_dense.tolist()

#deciders using the open vectorizer
#eodf = pd.DataFrame(eo_denselist)
eodf = pd.DataFrame(eo_denselist, columns=eo_feature_names, index=file_list)

# there is a lot of 0.00 so just pull out a few specific features

# compare deciders with open
counts = cosine_similarity(eo_df,eodf)
ec_open = pd.DataFrame(counts.T, index=file_list)
ec_open=ec_open.rename(columns={0: "open"})

## Sovereign Internet TF/IDF

#define the vectorizer, and the ngram range (one to three words)
es_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

#fit the corpus to the vectorizer, and get the feature names
es_vectors = es_vectorizer.fit_transform(sovereigns)
es_feature_names = es_vectorizer.get_feature_names()
es_dense = es_vectors.todense()
es_denselist = es_dense.tolist()

#put the resulting calculations into a df and use feature_names for column names
es_df = pd.DataFrame(es_denselist, columns=es_feature_names)

#use the same sovereign vectorizers i used
es_vectors = es_vectorizer.transform(deciders)
es_feature_names = es_vectorizer.get_feature_names()
es_dense = es_vectors.todense()
es_denselist = es_dense.tolist()

#deciders using the sovereign vectorizer
esdf = pd.DataFrame(es_denselist, columns=es_feature_names, index=file_list)

# compare deciders with sovereign
counts = cosine_similarity(es_df,esdf)
ec_sovereign = pd.DataFrame(counts.T, index=file_list)
ec_sovereign=ec_sovereign.rename(columns={0: "sovereign"})

# compare in one table
ec = pd.concat([ec_open, ec_sovereign], axis=1, sort=False)

#################
# Politics
#################

file_list = []
file_name = "un/output/po_deciders.csv"
with open(file_name) as f_input:
    file_list = f_input.read().split('\n')

## Run the Deciders Against the Corpus to get comparison
deciders = []
for one_file in file_list:
    decider = "un/data/"+one_file
    with open(decider) as f_input:
        deciders.append(f_input.read())

sovereigns = []
file_name = "un/output/po_sov_corpus.csv"
with open(file_name) as f_input:
    sovereigns = f_input.read().split('\n')

currents = []
file_name = "un/output/po_op_corpus.csv"
with open(file_name) as f_input:
    currents = f_input.read().split('\n')

## Open Internet TF/IDF

#define the vectorizer, and the ngram range (one to two words)
#3-gram would be nice but I keep running out of memory
po_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

#fit the corpus to the vectorizer, and get the feature names
po_vectors = po_vectorizer.fit_transform(currents)
po_feature_names = po_vectorizer.get_feature_names()
po_dense = po_vectors.todense()
po_denselist = po_dense.tolist()

#put the resulting calculations into a df and use feature_names for column names
po_df = pd.DataFrame(po_denselist, columns=po_feature_names)

#use the same open vectorizers i used for the deciders
po_vectors = po_vectorizer.transform(deciders)
po_feature_names = po_vectorizer.get_feature_names()
po_dense = po_vectors.todense()
po_denselist = po_dense.tolist()

#deciders using the open vectorizer
#eodf = pd.DataFrame(eo_denselist)
podf = pd.DataFrame(po_denselist, columns=po_feature_names, index=file_list)

# there is a lot of 0.00 so just pull out a few specific features

# compare deciders with open
counts = cosine_similarity(po_df,podf)
pl_open = pd.DataFrame(counts.T, index=file_list)
pl_open=pl_open.rename(columns={0: "open"})

## Sovereign Internet TF/IDF

#define the vectorizer, and the ngram range (one to three words)
ps_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

#fit the corpus to the vectorizer, and get the feature names
ps_vectors = ps_vectorizer.fit_transform(sovereigns)
ps_feature_names = ps_vectorizer.get_feature_names()
ps_dense = ps_vectors.todense()
ps_denselist = ps_dense.tolist()

#put the resulting calculations into a df and use feature_names for column names
ps_df = pd.DataFrame(ps_denselist, columns=ps_feature_names)

#use the same sovereign vectorizers i used
ps_vectors = ps_vectorizer.transform(deciders)
ps_feature_names = ps_vectorizer.get_feature_names()
ps_dense = ps_vectors.todense()
ps_denselist = ps_dense.tolist()

#deciders using the sovereign vectorizer
psdf = pd.DataFrame(ps_denselist, columns=ps_feature_names, index=file_list)

# compare deciders with sovereign
counts = cosine_similarity(ps_df,psdf)
pl_sovereign = pd.DataFrame(counts.T, index=file_list)
pl_sovereign=pl_sovereign.rename(columns={0: "sovereign"})

# compare in one table
pl = pd.concat([pl_open, pl_sovereign], axis=1, sort=False)

