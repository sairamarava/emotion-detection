from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
emotions=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("description", "")
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    return jsonify({"prediction": emotions[prediction]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

