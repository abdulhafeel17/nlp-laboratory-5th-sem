import nltk
from nltk.corpus import treebank
from nltk.tag import hmm
from nltk.tokenize import word_tokenize

# Download required resources
nltk.download('treebank')
nltk.download('punkt')
nltk.download('punkt_tab')

# Load tagged sentences from Treebank corpus
train_data = treebank.tagged_sents()

# Train the Hidden Markov Model (HMM) tagger
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# Get input from user
text = input("Enter a sentence: ")

# Tokenize sentence
tokens = word_tokenize(text)

# Perform POS tagging using HMM
tagged_words = hmm_tagger.tag(tokens)

# Display tokens
print("\nTokens:")
print(tokens)

# Display POS tags
print("\nHMM POS Tags:")
for word, tag in tagged_words:
    print(word, "->", tag)

# Display tag meanings
print("\nCommon Tag Meanings:")
print("NN  -> Noun")
print("NNS -> Plural Noun")
print("NNP -> Proper Noun")
print("VB  -> Verb (Base Form)")
print("VBD -> Verb (Past Tense)")
print("VBG -> Verb (Gerund)")
print("JJ  -> Adjective")
print("RB  -> Adverb")
print("PRP -> Pronoun")
print("DT  -> Determiner")

# Count tagged words
print("\nTotal Words:", len(tokens))