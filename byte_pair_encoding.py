import pandas as pd
import collections
import re


def get_stats(corpus):
    pairs = collections.defaultdict(int)
    for word, freq in corpus.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs


def merge_vocab(pair, corpus_in):
    corpus_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    for word in corpus_in:
        w_out = p.sub(''.join(pair), word)
        corpus_out[w_out] = corpus_in[word]
    
    return corpus_out

#reading .txt file
text = pd.read_csv("sample.txt",header=None)

#converting a dataframe into a single list 
corpus=[]
for row in text.values:
    tokens = row[0].split(" ")
    for token in tokens:
        corpus.append(token)

vocab = list(set(" ".join(corpus)))
vocab.remove(' ')

#split the word into characters
corpus = [" ".join(token) for token in corpus]

#appending </w>
corpus=[token+' </w>' for token in corpus]

#returns frequency of each word
corpus = collections.Counter(corpus)

#convert counter object to dictionary
corpus = dict(corpus)

print(get_stats(corpus))

pairs= get_stats(corpus)
best = max(pairs, key=pairs.get)
print("Most Frequent pair:",best)

num_merges = 100
for i in range(num_merges):
    corpus = merge_vocab(best, corpus)
    print("After Merging:", corpus)

    #convert a tuple to a string
    best = "".join(list(best))

    #append to merge list and vocabulary
    merges = []
    merges.append(best)
    vocab.append(best)
