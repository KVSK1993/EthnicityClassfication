# EthnicityClassfication Based on Full Name

Data Cleaning / preprocessing:
1.	Removed names which have either first name or last name missing.
2.	Capitalised the first letter of Last name and First name.
3.	Concatenated first name and last name to make a new column ‘full_name’ so as to make predictions based on full name.
4.	Divided the dataset, 80 % for training and 20% for testing using stratified sampling.
5.	Calculated tf-idf scores of bigrams, trigrams and four grams (character level) using TfidfVectorizer.


**Comments:**
The main difference between HashingVectorizer and TfidfVectorizer is that HashingVectorizer applies a hashing function to term frequency counts in each document, whereas TfidfVectorizer scales those term frequency counts in each document by penalising terms that appear more widely across the corpus.
Terms in our case are bigrams, trigrams and fourgrams at character level.

**References:**
1)	Vadehra, A., Grossman, M.R. and Cormack, G.V., 2017. Impact of Feature Selection on Micro-Text Classification. arXiv preprint arXiv:1708.08123.
2)	Sood, G. and Laohaprapanon, S., 2018. Predicting Race and Ethnicity From the Sequence of Characters in a Name. arXiv preprint arXiv:1805.02109.

