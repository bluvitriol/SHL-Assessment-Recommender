from flask import Flask, request, jsonify
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

if __name__ == "__main__":
    app.run(debug=True)



# app = Flask(__name__)
# CORS(app)

# # Load model and index
# # model = SentenceTransformer('all-MiniLM-L6-v2')
# # index = faiss.read_index("shl_faiss_index.index")

# # # Load metadata
# # with open("shl_metadata.pkl", "rb") as f:
# #     metadata = pickle.load(f)

# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"status": "healthy"}), 200

# @app.route("/recommend", methods=["POST"])
# def recommend():
#     try:
#         data = request.get_json()
#         query = data.get("query", "")
#         if not query:
#             return jsonify({"error": "Missing query parameter"}), 400

#         # embedding = model.encode([query])
#         # D, I = index.search(np.array(embedding), k=10)

#         # results = []
#         # for idx in I[0]:
#         #     if idx < len(metadata):
#         #         results.append(metadata[idx])

#         # return jsonify({"recommended assessments": results}), 200
#         results = recommend_assessments(query)  # now includes filtering logic

#         if not results:
#             return jsonify({"message": "No relevant assessments found"}), 200

#         return jsonify({"recommended assessments": results}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)
