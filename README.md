# Classifying the Digital Deciders

> <----Global and Open--------------------Sovereign and Controlled---->
>
> The `Digital Deciders` are somewhere in between...                           

This repository contains Jupyter notebooks, python code (.py), and data files used to classify "Digital Deciders" on how they view the future of the Internet.

## An analysis of the General Debate speeches at the United Nations General Assembly

There is currently a battle underway over who has control over the Internet. Is it a global, open resource, where everyone has access, or is it subject to each country's own laws? While some countries have staked out their positions (global and open vs sovereign and controlled), others have not yet publicly picked a side. 

We use natural language processing tools to try to classify these countries into the side with countries they are the most similar to.

### Methodology:
GitHub.io link

### Data:

Text files in the `data` directory, broken out by General Assembly session and year (##-YYYY). Each file has the file format (XXX_##_YYYY.txt) where the first three letters refer to the country code. The directory contains statements from representatives of member countries giving their position on world topics from 1970 to 2018. 

The original files are available on Harvard Dataverse as [The UN General Debate Corpus](https://data
verse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0TJX8Y).

### Code:

* Jupyter notebooks to create the corpus (*part1_corpus.ipynb*)
* Python code to run the similarity comparison (*part2_compare.py*)
* Python code to create the classifier (*part3_classify.py*)

Story:
GitHub.io link (not yet)


