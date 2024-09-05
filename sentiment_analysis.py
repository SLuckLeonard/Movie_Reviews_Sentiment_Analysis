import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
import joblib

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('data/IMDB Dataset.csv')

# Basic data preprocessing
df['review'] = df['review'].str.lower()  # Convert to lowercase
df['review'] = df['review'].str.replace(r'[^\w\s]', '')  # Remove punctuation

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.3, random_state=42)

# Initialize CountVectorizer
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))

# Fit and transform training data
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Initialize Naive Bayes model and fit to the training data
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Make predictions on the test data
predictions = model.predict(X_test_vect)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
