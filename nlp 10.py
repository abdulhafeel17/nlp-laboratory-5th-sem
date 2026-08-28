from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

docs = []
labels = []

n = int(input(&quot;Enter number of documents: &quot;))

for i in range(n):
docs.append(input(&quot;Enter document: &quot;))
labels.append(input(&quot;Enter category: &quot;))

rule_pred = []

for doc in docs:

doc = doc.lower()

if &quot;contract&quot; in doc:
rule_pred.append(&quot;contract&quot;)
elif &quot;judgment&quot; in doc:
rule_pred.append(&quot;judgment&quot;)
else:
rule_pred.append(&quot;agreement&quot;)

rule_acc = accuracy_score(labels, rule_pred)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

ml_pred = model.predict(X)

ml_acc = accuracy_score(labels, ml_pred)

print(&quot;\nRule-Based Accuracy:&quot;, rule_acc)
print(&quot;Maximum Entropy Accuracy:&quot;, ml_acc)