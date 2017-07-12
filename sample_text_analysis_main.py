# This file is for the Python NLP analysis of the text documents
# within R terminal, run as:
# system("python3 sample_text_analysis_main.py")

# import relevant packages
# NB. All of these packages must be installed in order to run this script
# See https://packaging.python.org/installing/ for help
import nltk, os, re, sys
import pandas as pd
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk import ngrams
from collections import Counter
import string
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed

# -------------------- word search
# this function takes a word and a corpus
# and a returns a list of 0/1 for each
def find_term_intext(term, text, verbose = False):
	# if corp is a nltk TEXT object
	term_yn = []
	jobs = len(text)
	inc = round(jobs / 50)
	for j,s in enumerate(text):
		result = re.search('\\b'+term+'\\b', s)
		term_yn.append(1 if result else 0)
	if (verbose and j%inc==0): print('Progress:',j,'/',jobs)
	return term_yn

# prepocess the text here
def preprocess(t):
	rem_chars = "[!\"#$%&()*+,:;<=>?@[\\]^_`{|}~0123456789]" # remove these
	rep_chars = "[-./\']" # replace these
	t_temp = re.sub(rem_chars, "", t.lower())
	t_temp = re.sub(rep_chars, " ", t_temp)
	t_strip_lower_filt = [w for w in t_temp.split() if not w in stopwords.words('english')]
	return " ".join(t_strip_lower_filt)

# load the data
corpusdir = 'corpus_txt/' # Directory of corpus.
mycorp_raw = PlaintextCorpusReader(corpusdir, '.*')
file_index = mycorp_raw.fileids()

# preprocess the text (slow)
# uncomment one of the following lines for usual vs parallel processing
#mycorp_proc = nltk.Text([preprocess(mycorp_raw.raw(f)) for f in file_index])
mycorp_proc = Parallel(n_jobs=3,verbose=True)(delayed(preprocess)(mycorp_raw.raw(f)) for f in file_index)


# get ngrams (1-3)
vectorizer_ngrams = CountVectorizer(min_df = 0.05, ngram_range=(1, 3))
mat_ngrams = vectorizer_ngrams.fit_transform(mycorp_proc)
n_df = pd.DataFrame(data = mat_ngrams.A, 
	columns = vectorizer_ngrams.get_feature_names())
n_df['pt_id'] = [i[:-4] for i in file_index]
# write results to file
n_df.to_csv('ngrams_dtm.csv', index = False)

# note the following analysis is redundant in this small case, since all words 
# will be captured. In a very large dataset we might limit the capture of n-grams
# to those that only occur in say 5% fo the documents, thus potentially excluding other
# terms of interest such as those below
# choose our list of terms of interest
some_terms = ['comfort care', 'wishes', 'goals of care', 'family', "wouldn't want",
"doesn't want", 'poor prognosis', 'high risk', 'code status',
'family meeting', 'hospice', 'palliative', 'palliation', 'palliative care',
'pall care', 'death', 'die', 'survival', 'mortal', 'mortality', 'comfortable']

# now make a data frame so we can export these findings
df_results = pd.DataFrame.from_items([('pt_id', [i[:-4] for i in file_index])])
# now loop through terms and build results
for tt in some_terms:
	df_results[tt] = find_term_intext(preprocess(tt), mycorp_proc)
# save results to file so we can analyze back in R
df_results.to_csv('text_results_keyterms.csv', index = False)