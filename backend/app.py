from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import pickle

# ------------------------------------
# 🔑 NGROK AUTH TOKEN (PUT HERE)
# ------------------------------------
ngrok.set_auth_token("35usUDOJZduCSr1MxoxmaOYVM0H_7ixfvcr6SvrQ7dN3AehCK")

# Kill old tunnels to avoid ERR_NGROK_334
ngrok.kill()

# Start ngrok tunnel on port 5000
public_url = ngrok.connect(5000)
print("🌍 Public URL:", public_url)

# ------------------------------------
# 🚀 FLASK APP
# ------------------------------------
app = Flask(__name__)
CORS(app)   # allow frontend → backend requests

# ------------------------------------
# 📦 LOAD MODEL
# ------------------------------------
with open("../model/model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# ------------------------------------
# 🔮 PREDICTION API
# ------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Message is required"}), 400

    text = data["message"]

    text_vec = vectorizer.transform([text])

    # Prediction
    prediction = model.predict(text_vec)[0]

    # Probabilities
    proba = model.predict_proba(text_vec)[0]

    # Match confidence to predicted class
    if prediction == "spam":
        confidence = proba[1] 
    else:
        confidence = proba[0]

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 2)
    })



# ------------------------------------
# ▶️ RUN SERVER
# ------------------------------------
if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)
