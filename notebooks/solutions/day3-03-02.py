
test_encoded = vectorizer.transform(test_sentences).toarray()
regressor.predict(test_encoded)

