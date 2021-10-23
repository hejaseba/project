import nltk, re, pprint, string

from nltk import word_tokenize, sent_tokenize

string.punctuation = string.punctuation +'“'+'”'+'-'+'’'+'‘'+'—'

string.punctuation = string.punctuation.replace('.', '')
file = open('corpus.txt', encoding = 'utf8').read()



#preprocess data , removing special characters and new lines
file_nl_removed = ""
for line in file:
  line_nl_removed = line.replace("\n", " ")      #removes newlines
  file_nl_removed += line_nl_removed
file_p = "".join([char for char in file_nl_removed if char not in string.punctuation])   #removes all special characters


# sentences
sents = nltk.sent_tokenize(file_p)
print("The number of sentences is", len(sents)) 


# tokens / words
words = nltk.word_tokenize(file_p)
print("The number of tokens is", len(words)) 


# average number of tokens per sentence
average_tokens = round(len(words)/len(sents))
print("The average number of tokens per sentence is",average_tokens) 

# unique tokens ( vocabulary )
unique_tokens = set(words)
print("The number of unique tokens are", len(unique_tokens)) 

# point is in unique_tokens and in words ( tokens )

####################

# points ( 0 )
# build n -grams ( 1 )
# frequency of n-gram ; start and end word ; point ; how to evaluate ; uses of n - gram ( 2 )
# chapter 3 ; smoothing ( 3 )
# use of language models ; language models in sentiment analysis for example ( 4 )
# wall street journal in stanford book data set , evaluate perplexity ( 5 )

####################

from nltk.util import ngrams
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


# unigram , bigram , trigram , and fourgram language models

unigram=[]
bigram=[]
trigram=[]
fourgram=[]
tokenized_text = []

for sentence in sents:
    sentence = sentence.lower()
    sequence = word_tokenize(sentence)
    for word in sequence:
        if(word=='.'):
            sequence.remove(word)
        else:
            unigram.append(word)
    tokenized_text.append(sequence)
    bigram.extend(list(ngrams(sequence,2)))
    trigram.extend(list(ngrams(sequence, 3)))
    fourgram.extend(list(ngrams(sequence, 4)))
    
