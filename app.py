from flask import Flask, render_template, request
import pickle
import nltk

from nltk.corpus import stopwords

# Download stopwords (safe if already exists)
nltk.download('stopwords')

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        email = request.form.get("email")

        cleaned = clean_text(email)
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)

        prediction = "🚫 Spam" if result[0] == 1 else "✅ Ham"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
