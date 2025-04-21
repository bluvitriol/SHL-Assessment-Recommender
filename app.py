from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from flask_cors import CORS
from recommend_assessments import recommend_assessments


app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        query = data.get("query", "")
        if not query:
            return jsonify({"error": "Missing query parameter"}), 400

        results = recommend_assessments(query)

        if not results:
            return jsonify({"message": "No relevant assessments found"}), 200

        return jsonify({"recommended assessments": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)


