# # # # # # # # # # # # # # # # # # # #
# Classifying the Digital Deciders:
# The Choice Between an Open Internet and a Sovereign One
# Fahmida Y Rashid (fr48)
# Part 3: Using a Classifier
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
## Part Three: Creating a classifer

I am going to use the Naive Bayes text classifier to see if I can train the system to predict 
where Digital Deciders should go

* Read in the relevant speeches from sovereign and open groups and tag them as open or sovereign
* tokenize by sentence
* Hold back some of the speeches for testing
'''

# import nltk.classify.util
# from nltk.classify import NaiveBayesClassifier

import pandas as pd
from nltk import sent_tokenize
from textblob.classifiers import NaiveBayesClassifier
from sklearn.model_selection import train_test_split

file_list = []
file_name = "un/data/unga_speeches.txt"
with open(file_name) as f_input:
    file_list = f_input.read().split('\n')

# New America definitions
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

sovereigns = []
currents = []
deciders = []
unknowns = []

for one_file in file_list:
    if (one_file[8:11]) in global_open:
        currents.append(one_file)
    elif (one_file[8:11]) in sovereign_controlled:
        sovereigns.append(one_file)
    elif (one_file[8:11]) in digital:
        deciders.append(one_file)
    else:
        unknowns.append(one_file)

sov_files = []

for sovereign in sovereigns:
    sovereign = 'un/data/' + sovereign
    with open(sovereign) as f_input:
        sov_files.append(f_input.read())

new_sov = []
for speech in sov_files:
    sentences = sent_tokenize(speech)
    for sentence in sentences:
        new_sov.append(sentence)

#tag each sentence as 'SOV'
new_sov = pd.DataFrame(new_sov)
new_sov =new_sov.rename(columns={0: "sentences"})

new_sov['group']='SOV'

open_files = []

for current in currents:
    current = 'un/data/' + current
    with open(current) as f_input:
        open_files.append(f_input.read())

new_open = []
for speech in open_files:
    sentences = sent_tokenize(speech)
    for sentence in sentences:
        new_open.append(sentence)

#tag each sentence as 'OPN'
new_open = pd.DataFrame(new_open)
new_open =new_open.rename(columns={0: "sentences"})

new_open['group']='OPN'

speeches = pd.concat([new_open, new_sov], axis=0, sort=False)

# Decider files
decider_files = []

for decider in deciders:
    decider = 'un/data/' + decider
    with open(decider) as f_input:
        decider_files.append(f_input.read())

new_decider = []
for speech in decider_files:
    sentences = sent_tokenize(speech)
    for sentence in sentences:
        new_decider.append(sentence)

# create classifier
train, test = train_test_split(speeches, test_size=0.1)

train_set = list(train.itertuples(index=False, name=None))
test_set = list(test.itertuples(index=False, name=None))

my_classifier = NaiveBayesClassifier(train_set)

#try out the deciders
my_classifier.classify(new_decider[0])
prob_dist = my_classifier.prob_classify(new_decider[1])
print(prob_dist.max())
print(round(prob_dist.prob('SOV'), 2))
print(round(prob_dist.prob('OPN'),2))

#test accuracy
accuracy=my_classifier.accuracy(test_set)
print(accuracy)
informative=my_classifier.show_informative_features(5)