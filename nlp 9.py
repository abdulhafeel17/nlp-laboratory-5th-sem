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

Output:

Result:
Legal documents were successfully classified using Rule-Based and Maximum Entropy
classifiers. The classification accuracies were calculated and compared.

Expt.No:10 UTILIZE WORD AND PHRASE-BASED CLUSTERING
ALGORITHMS TO IDENTIFY PATTERNS IN SOCIAL MEDIA
CONVERSATIONS AND ANALYZE THEIR IMPLICATIONS
FOR MARKETING STRATEGIES..

Date:

Aim
To apply clustering techniques on social media posts using TF-IDF (Term Frequency–Inverse
Document Frequency) and K-Means to identify customer trends and marketing insights.
Procedure
1. Import the required Python libraries such as Scikit-learn.
2. Accept social media posts as input from the user.
3. Convert all posts into lowercase format.
4. Remove stopwords and unnecessary symbols from the text.
5. Extract words and phrases using TF-IDF vectorization.
6. Generate unigram and bigram features from the posts.
7. Apply the K-Means clustering algorithm.
8. Group similar social media posts into clusters.
9. Assign cluster labels to each post.
10. Display the clustered posts.
11. Extract important keywords and phrases from each cluster.
12. Analyze customer opinions and trends based on clusters.
13. Display marketing insights from the clustered data.
14. Evaluate how clustering helps identify customer interests and issues.
15. Conclude the usefulness of clustering for social media marketing analysis.
Program:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

posts = []

n = int(input(&quot;Enter number of posts: &quot;))

for i in range(n):
post = input(&quot;Enter post: &quot;)
posts.append(post)

k = int(input(&quot;Enter number of clusters: &quot;))

vectorizer = TfidfVectorizer(
stop_words=&#39;english&#39;,
ngram_range=(1,2)
)

X = vectorizer.fit_transform(posts)

model = KMeans(
n_clusters=k,
random_state=42,
n_init=10
)

model.fit(X)

labels = model.labels_

print(&quot;\nCluster Results:\n&quot;)

for i in range(len(posts)):
print(&quot;Post:&quot;, posts[i])
print(&quot;Cluster:&quot;, labels[i])
print()

terms = vectorizer.get_feature_names_out()

print(&quot;Important Keywords:\n&quot;)

for i in range(k):

center = model.cluster_centers_[i]
top = center.argsort()[-5:]

print(&quot;Cluster&quot;, i)

for j in top:
print(terms[j])

print()

print(&quot;Marketing Insight:&quot;)
print(&quot;Similar customer opinions are grouped together.&quot;)
print(&quot;Clusters help identify product trends and issues.&quot;)

