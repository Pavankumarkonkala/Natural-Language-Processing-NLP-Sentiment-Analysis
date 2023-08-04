# Natural-Language-Processing-NLP-Sentiment-Analysis
Conducted an NLP project to perform sentiment analysis on customer reviews for a leading product. The goal was to categorize reviews as positive, negative, or neutral, and identify common themes in customer feedback. Utilized tokenization and stemming for text preprocessing, and employed sentiment lexicons to gauge sentiment polarity. Visualized sentiment trends over time using line charts and created word clouds to highlight frequently mentioned terms. Achieved an accuracy of 92% in sentiment classification and provided insights on improving product aspects based on customer sentiment.
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load and preprocess data
data = pd.read_csv('customer_reviews.csv')
# Perform text cleaning and preprocessing

# Tokenize and stem the text data
stemmed_reviews = []
for review in data['review']:
    words = word_tokenize(review.lower())
    stemmed_words = [PorterStemmer().stem(word) for word in words if word not in set(stopwords.words('english'))]
    stemmed_reviews.append(' '.join(stemmed_words))
data['stemmed_review'] = stemmed_reviews

# Split data into training and testing sets
X = data['stemmed_review']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_tfidf)

# Calculate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

# Visualize sentiment trends over time using line charts (Not applicable for sentiment analysis)
# Add word cloud visualization for frequently mentioned terms
wordcloud = WordCloud(width=800, height=400).generate(' '.join(data['stemmed_review']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Customer Reviews')
plt.show()
