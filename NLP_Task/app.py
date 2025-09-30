from flask import Flask, request, jsonify
import joblib


model = joblib.load("sentiment_model.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  
    tweet = data.get("text", "")

    if not tweet.strip():
        return jsonify({"error": "No text provided"}), 400

  
    prediction = model.predict([tweet])[0]


    label_map = {0: "negative", 1: "positive"}
    sentiment = label_map.get(prediction, str(prediction))

    return jsonify({"tweet": tweet, "prediction": int(prediction), "sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)
