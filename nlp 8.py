import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

reviews = []
n = int(input(&quot;Enter number of reviews: &quot;))

for i in range(n):
reviews.append(input(&quot;Enter review: &quot;))

vectorizer = CountVectorizer(stop_words=&#39;english&#39;)
X = vectorizer.fit_transform(reviews)

lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

words = vectorizer.get_feature_names_out()

print(&quot;\nTopics:&quot;)
for i, topic in enumerate(lda.components_):
print(&quot;\nTopic&quot;, i + 1)
top_words = topic.argsort()[-5:]
for j in top_words:
print(words[j])

X_dense = X.toarray()

tsne = TSNE(n_components=2, random_state=42, perplexity=2)
X_tsne = tsne.fit_transform(X_dense)

print(&quot;\nt-SNE Coordinates:&quot;)
for i, point in enumerate(X_tsne):
print(&quot;Review&quot;, i + 1, &quot;:&quot;, point)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

for i in range(len(reviews)):
plt.text(X_tsne[i, 0], X_tsne[i, 1], &quot;R&quot; + str(i + 1))

plt.title(&quot;t-SNE Visualization of Customer Reviews&quot;)
plt.xlabel(&quot;Dimension 1&quot;)
plt.ylabel(&quot;Dimension 2&quot;)
plt.show()