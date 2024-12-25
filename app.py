import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = "d:\CODING\My Codes\sysnnetsec\malicious_phish.csv"  
data = pd.read_csv(file_path, names=['url', 'type']) 

data.rename(columns={'url': 'URL', 'type': 'Label'}, inplace=True)

data['URL'] = data['URL'].str.lower()  
data.dropna(inplace=True)  
data.drop_duplicates(inplace=True)  

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])

print("Label Encoding Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))  # Character-level n-grams
X = vectorizer.fit_transform(data['URL'])
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

def predict_url(url):
    url = url.lower()  
    vectorized_url = vectorizer.transform([url])
    prediction = clf.predict(vectorized_url)[0]
    return label_encoder.inverse_transform([prediction])[0]

test_urls = [
    "http://example.com",
    "http://malicious-example.com/phishing"
]
for url in test_urls:
    print(f"URL: {url} -> Prediction: {predict_url(url)}")
