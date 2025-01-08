import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns  # Bu satırı kaldırıyoruz

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# NLTK Veri İndirme
nltk.download('stopwords')
nltk.download('wordnet')

# Veri Yükleme Fonksiyonu
def load_data(spam_dir, ham_dir):
    data = []
    
    # Spam e-postalarını yükleme
    for filename in os.listdir(spam_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(spam_dir, filename), 'r', encoding='latin-1') as file:
                content = file.read()
                data.append([content, 'spam'])
                
    # Ham e-postalarını yükleme
    for filename in os.listdir(ham_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(ham_dir, filename), 'r', encoding='latin-1') as file:
                content = file.read()
                data.append([content, 'ham'])
    
    return pd.DataFrame(data, columns=['text', 'label'])

# Klasör yolları belirtilir
spam_directory = 'enron1/spam'
ham_directory = 'enron1/ham'

# Yol doğrulama
if not os.path.exists(spam_directory):
    raise FileNotFoundError(f"Spam klasörü bulunamadı: {spam_directory}")
if not os.path.exists(ham_directory):
    raise FileNotFoundError(f"Ham klasörü bulunamadı: {ham_directory}")

# Veri setini yükleyin
df = load_data(spam_directory, ham_directory)

# İlk birkaç satırı görüntüleme
print(df.head())

# Etiket dağılımını görselleştirme
label_counts = df['label'].value_counts()
plt.figure(figsize=(6,4))
plt.bar(label_counts.index, label_counts.values, color=['skyblue', 'salmon'])
plt.title('Spam vs Ham')
plt.xlabel('Etiket')
plt.ylabel('Sayısı')
plt.xticks(rotation=0)
plt.show()

# Metin Temizleme ve Ön İşleme
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Küçük harfe çevirme
    text = text.lower()
    # HTML etiketlerini kaldırma
    text = re.sub(r'<[^>]+>', ' ', text)
    # Özel karakterleri kaldırma
    text = re.sub(r'\W', ' ', text)
    # Tekrarlayan boşlukları tek boşluk yapma
    text = re.sub(r'\s+', ' ', text).strip()
    # Sayıları kaldırma
    text = re.sub(r'\d', ' ', text)
    # Tokenizasyon
    tokens = text.split()
    # Stopwords kaldırma ve lemmatizasyon
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Tekrar birleştirme
    return ' '.join(tokens)

# Ön işleme uygulama
print("Metin temizleme ve ön işleme başladı...")
df['cleaned_text'] = df['text'].apply(preprocess_text)
print("Metin temizleme tamamlandı.")



# Verinin Bölünmesi
X = df['cleaned_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Eğitim seti boyutu: {X_train.shape[0]}")
print(f"Test seti boyutu: {X_test.shape[0]}")

# TF-IDF vektörleştiriciyi N-gram'larla güncelleme
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Örneğin, e-postaların karakter sayısını bir özellik olarak ekleyebiliriz
df['char_count'] = df['cleaned_text'].apply(len)

# TF-IDF ve char_count'u birleştirme
from scipy.sparse import hstack

X = hstack([X_train_tfidf, df.loc[X_train.index, 'char_count'].values.reshape(-1,1)])
X_test = hstack([X_test_tfidf, df.loc[X_test.index, 'char_count'].values.reshape(-1,1)])

# Model Eğitimi
model = MultinomialNB()
model.fit(X, y_train)

# Model Değerlendirmesi
y_pred = model.predict(X_test)

# Değerlendirme raporu
print("Model Değerlendirme Raporu (Ek Özellikler):")
print(classification_report(y_test, y_pred))

# Modeli Kaydetme
import joblib
joblib.dump(model, 'spam_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model ve vektörleştirici başarıyla kaydedildi.")

# Karışıklık Matrisi
conf_mat = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Karışıklık Matrisi')
plt.colorbar()
tick_marks = np.arange(len(['Ham', 'Spam']))
plt.xticks(tick_marks, ['Ham', 'Spam'], rotation=45)
plt.yticks(tick_marks, ['Ham', 'Spam'])

# Her hücreye değer yazma
thresh = conf_mat.max() / 2.
for i in range(conf_mat.shape[0]):
    for j in range(conf_mat.shape[1]):
        plt.text(j, i, format(conf_mat[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_mat[i, j] > thresh else "black")

plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.tight_layout()
plt.show()

# Hiperparametre Ayarlamaları (Opsiyonel)
from sklearn.model_selection import GridSearchCV

parameters = {
    'alpha': [0.1, 0.5, 1.0],
    'fit_prior': [True, False]
}

grid_search = GridSearchCV(MultinomialNB(), param_grid=parameters, cv=5, scoring='f1')
grid_search.fit(X, y_train)  # Dengelenmiş veri kullanılır

print(f"En iyi parametreler: {grid_search.best_params_}")
print(f"En iyi F1 skoru: {grid_search.best_score_}")

# En iyi modeli kullanma
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print("En iyi Naive Bayes Modelinin Değerlendirme Raporu:")
print(classification_report(y_test, y_pred_best))

# Farklı Modeller Deneme (Lojistik Regresyon ve SVM)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Lojistik Regresyon
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X, y_train)
y_pred_lr = lr_model.predict(X_test)
print("Lojistik Regresyon:")
print(classification_report(y_test, y_pred_lr))

# SVM
svm_model = LinearSVC()
svm_model.fit(X, y_train)
y_pred_svm = svm_model.predict(X_test)
print("SVM:")
print(classification_report(y_test, y_pred_svm))

# Sonuçları Karşılaştırma
models = {
    'Naive Bayes': classification_report(y_test, y_pred, output_dict=True),
    'En İyi Naive Bayes': classification_report(y_test, y_pred_best, output_dict=True),
    'Lojistik Regresyon': classification_report(y_test, y_pred_lr, output_dict=True),
    'SVM': classification_report(y_test, y_pred_svm, output_dict=True)
}

results = pd.DataFrame({
    model: metrics['f1-score'] for model, metrics in models.items()
}, index=['F1 Skoru']).T

print("Modellerin F1 Skorlarının Karşılaştırılması:")
print(results)


