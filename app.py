from flask import Flask, request, render_template, redirect, url_for, flash
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import imaplib
import email
from email.header import decode_header
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)
app.secret_key = 'secret_key'

# Model ve vektörleştiriciyi yükleme
try:
    model = joblib.load('spam_classifier.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Model ve vektörleştirici başarıyla yüklendi.")
except FileNotFoundError as e:
    print(f"Hata: {e}")
    print(
        "Model veya vektörleştirici dosyaları bulunamadı. Lütfen project.py dosyasını çalıştırarak .pkl dosyalarını oluşturun.")
    exit(1)

# NLTK Veri İndirme
nltk.download('stopwords')
nltk.download('wordnet')

# Metin temizleme fonksiyonu
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)  # HTML etiketlerini kaldırma
    text = re.sub(r'\W', ' ', text)  # Özel karakterleri kaldırma
    text = re.sub(r'\s+', ' ', text).strip()  # Tekrarlayan boşlukları tek boşluk yapma
    text = re.sub(r'\d', ' ', text)  # Sayıları kaldırma
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/classify')
def classify():
    return render_template('classify.html')


@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    cleaned_email = preprocess_text(email_text)
    email_tfidf = vectorizer.transform([cleaned_email])

    # Karakter sayısını ekleme
    char_count = len(cleaned_email)
    email_features = hstack([email_tfidf, csr_matrix([[char_count]])])

    prediction = model.predict(email_features)[0]
    proba = max(model.predict_proba(email_features)[0])

    return render_template('result.html', prediction=prediction, proba=proba)


@app.route('/email_login')
def email_login():
    return render_template('email_login.html')


@app.route('/classify_emails', methods=['POST'])
def classify_emails():
    user_email = request.form['user_email']
    user_password = request.form['user_password']

    # Gmail için IMAP sunucusu, farklı sağlayıcılar için değiştirilmelidir
    imap_server = 'imap.gmail.com'
    try:
        # IMAP sunucusuna bağlanma
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(user_email, user_password)
        mail.select("inbox")  # Gelen kutusunu seçme

        # Son 10 e-postayı çekme
        result, data = mail.search(None, "ALL")
        email_ids = data[0].split()
        latest_email_ids = email_ids[-10:]  # Son 10 e-posta

        emails = []
        for eid in latest_email_ids:
            res, msg_data = mail.fetch(eid, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    # E-posta başlıklarını alma
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else 'utf-8')
                    from_, encoding = decode_header(msg.get("From"))[0]
                    if isinstance(from_, bytes):
                        from_ = from_.decode(encoding if encoding else 'utf-8')

                    # E-posta içeriğini alma
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get("Content-Disposition"))
                            try:
                                body = part.get_payload(decode=True).decode()
                            except:
                                body = ""
                            if content_type == "text/plain" and "attachment" not in content_disposition:
                                break
                    else:
                        content_type = msg.get_content_type()
                        body = msg.get_payload(decode=True).decode()

                    # Metni temizleme
                    cleaned_body = preprocess_text(body)
                    emails.append({
                        'subject': subject,
                        'from': from_,
                        'body': cleaned_body
                    })
        mail.logout()

        # E-postaları sınıflandırma
        classifications = []
        for email_data in emails:
            email_tfidf = vectorizer.transform([email_data['body']])
            char_count = len(email_data['body'])
            email_features = hstack([email_tfidf, csr_matrix([[char_count]])])
            prediction = model.predict(email_features)[0]
            proba = max(model.predict_proba(email_features)[0])
            classifications.append({
                'subject': email_data['subject'],
                'from': email_data['from'],
                'prediction': prediction,
                'proba': proba
            })

        return render_template('email_results.html', classifications=classifications, user_email=user_email)

    except imaplib.IMAP4.error:
        flash("E-posta adresi veya şifre hatalı!", "danger")
        return redirect(url_for('email_login'))
    except Exception as e:
        flash(f"Bir hata oluştu: {str(e)}", "danger")
        return redirect(url_for('email_login'))


if __name__ == '__main__':
    app.run(debug=True)
