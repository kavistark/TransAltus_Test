
import pandas as pd
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    words = text.split()
    filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return ' '.join(filtered_words)

def train_model(df):
    df['cleaned_text'] = df['review_text'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    
    return model, vectorizer

def predict_category(model, vectorizer, new_review):
    cleaned = preprocess_text(new_review)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return prediction[0]

if __name__ == "__main__":
    # Load dataset
    df = load_dataset("reviews.csv")
    
    # Train model
    model, vectorizer = train_model(df)
    
    # Test predictions
    test_reviews = [
        "The product was damaged and feels very cheap in quality.",
        "My order never arrived and no one responded to my emails.",
        "Fantastic quality and fast delivery!",
        "Support team helped me within 10 minutes."
    ]

    print("\n--- Test Predictions ---")
    for review in test_reviews:
        result = predict_category(model, vectorizer, review)
        print(f"Review: {review}\nPredicted Category: {result}\n")
