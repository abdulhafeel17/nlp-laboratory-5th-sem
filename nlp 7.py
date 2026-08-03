import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.tag import hmm
from nltk.corpus import treebank

# Download required resources (only first time)
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("treebank")

# -----------------------------
# Input Tweet
# -----------------------------
tweet = input("Enter a tweet: ")

# Convert to lowercase and tokenize
tokens = nltk.word_tokenize(tweet.lower())

print("\nTokens:")
print(tokens)

# -----------------------------
# N-GRAM MODEL
# -----------------------------
print("\n========== N-GRAM MODEL ==========")

# Generate n-grams
unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

print("\nUnigrams:")
for gram in unigrams:
    print(gram)

print("\nBigrams:")
for gram in bigrams:
    print(gram)

print("\nTrigrams:")
for gram in trigrams:
    print(gram)

# Word Frequencies
print("\nWord Frequencies:")
fd = FreqDist(tokens)

for word, freq in fd.items():
    print(f"{word} : {freq}")

# -----------------------------
# HMM MODEL
# -----------------------------
print("\n========== HMM MODEL ==========")

# Train HMM using Treebank tagged sentences
train_data = treebank.tagged_sents()[:3000]

trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train(train_data)

# Predict POS tags
tagged = tagger.tag(tokens)

print("\nPredicted POS Tags:")
for word, tag in tagged:
    print(f"{word} --> {tag}")

# -----------------------------
# COMPARISON
# -----------------------------
print("\n========== COMPARISON ==========")

print("\nN-Gram Model")
print("• Learns word sequences.")
print("• Generates unigram, bigram, and trigram patterns.")
print("• Used in text prediction and language modeling.")

print("\nHidden Markov Model (HMM)")
print("• Predicts Part-of-Speech (POS) tags.")
print("• Uses transition and emission probabilities.")
print("• Used in sequence labeling and NLP tasks.")

print("\nConclusion:")
print("N-Gram captures local word sequences, while HMM captures contextual relationships through probabilistic state transitions.")