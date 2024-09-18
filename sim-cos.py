from sklearn.metrics.pairwise import cosine_similarity
X = [[1, 128, 128, 16], [3, 3, 16, 1], [1, 1, 1, 16], [1, 128, 128, 16]]
Y = [[1, 16, 16, 184], [3, 3, 184, 1], [1, 1, 1, 184], [1, 16, 16, 184]]
print(cosine_similarity(X, X))
print(cosine_similarity(X, Y))
X = [[1, 14, 14, 96], [1, 1, 96, 576], [1, 1, 1, 576], [1, 14, 14, 576]]
Y = [[1, 112, 112, 16], [1, 1, 16, 96], [1, 1, 1, 96], [1, 112, 112, 96]]
print(cosine_similarity(X, Y))
