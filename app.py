from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return """
    <html>
        <head>
            <title>Emotion Detection</title>
        </head>
        <body>
            <h1>Emotion Detection</h1>
            <form action="/predict" method="post">
                <label for="description">Enter Text:</label><br>
                <textarea name="description" rows="4" cols="50"></textarea><br>
                <button type="submit">Detect Emotion</button>
            </form>
        </body>
    </html>
    """

@app.route("/predict", methods=["POST"])
def predict():
    # Get input text
    text = request.form["description"]
    
    # Transform text and predict
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    emotions=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    
    # Return the prediction
    return f"""
    <html>
        <head>
            <title>Emotion Result</title>
        </head>
        <body>
            <h1>Emotion Detection Result</h1>
            <p>Input: {text}</p>
            <p>Detected Emotion: {emotions[prediction]}</p>
            <a href="/">Go Back</a>
        </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)
