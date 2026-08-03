import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")

# Keywords indicating biomedical relations
keywords = ["treats", "reduces", "controls", "helps", "causes", "prevents", "cures"]

# User Input
sentence = input("Enter biomedical sentence: ")
actual = int(input("Enter Actual Relation (1 = Yes, 0 = No): "))

# Convert to lowercase and tokenize
tokens = word_tokenize(sentence.lower())

print("\nTokens:")
print(tokens)

# Predict relation
predicted = 0

for word in tokens:
    if word in keywords:
        predicted = 1
        break

print("\nPredicted Relation:", predicted)

# Evaluation
y_true = [actual]
y_pred = [predicted]

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\nEvaluation Results")
print("-------------------------")
print("Precision :", precision)
print("Recall    :", recall)
print("F1-Score  :", f1)

# Final Result
if predicted == actual:
    print("\nPrediction Correct")
else:
    print("\nPrediction Incorrect")