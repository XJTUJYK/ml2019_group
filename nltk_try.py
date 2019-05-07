import nltk

attr=list(set(nltk.corpus.brown.tagged_words()[:]))[1]
print(attr)
print(len(attr))
