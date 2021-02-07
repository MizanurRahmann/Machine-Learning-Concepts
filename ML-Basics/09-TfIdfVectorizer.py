from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(["I love machine learning and clustering algorithms",
                          "Apple, oranges and any kind of fruits are healthy",
                          "Is is fessible with machine learning algorithm?",
                          "My family is happy because of the healthy fruits"])
# print(tfidf.A)
print((tfidf * tfidf.T).A)