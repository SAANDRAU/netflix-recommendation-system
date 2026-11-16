
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import re

# ==========================================
# 1. INITIALIZE FLASK
# ==========================================

app = Flask(__name__)
from flask_cors import CORS
CORS(app)


# ==========================================
# 2. LOAD YOUR EXISTING MODEL
# ==========================================

# MODEL_PATH = "artifacts/Netfkix_Recommendation.keras"
# DATA_PATH = "artifacts/netflix_data.pkl"

# Get the directory containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define artifacts folder
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")


MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "Netflix_Recommendation.h5"
)


DATA_PATH = os.path.join(
    ARTIFACT_DIR,
    "netflix_data.pkl"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Netflix model not found. Run the notebook export cell first.")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Netflix dataset not found. Run the notebook export cell first.")

model = tf.keras.models.load_model(MODEL_PATH)

df = pd.read_pickle(DATA_PATH)

print("Netflix model loaded successfully!")
print("Dataset loaded successfully!")
print("Total titles:", len(df))


# ==========================================
# 3. FIND THE SELECTED TITLE
# ==========================================

def find_content(content_title):

    content_title = content_title.strip()

    if not content_title:
        return None

    # Exact title match
    exact_match = df[
        df["Title"].str.lower() == content_title.lower()
    ]

    if not exact_match.empty:
        return exact_match.iloc[0]

    # Partial title match
    partial_match = df[
        df["Title"].str.contains(
            re.escape(content_title),
            case=False,
            na=False,
            regex=True
        )
    ]

    if not partial_match.empty:
        return partial_match.iloc[0]

    return None


# ==========================================
# 4. YOUR RECOMMENDATION FUNCTION
# ==========================================

def recommend_similar(content_title, top_k=5):

    content_row = find_content(content_title)

    if content_row is None:
        return None, None

    content_id = int(content_row["Content_ID"])

    language_id = int(content_row["Language_ID"])

    content_type_id = int(content_row["Content_Type_ID"])

    # Make prediction using your trained model

    predictions = model.predict(
        {
            "content_id": np.array([content_id], dtype=np.int32),

            "language_id": np.array([language_id], dtype=np.int32),

            "content_type": np.array([content_type_id], dtype=np.int32)
        },
        verbose=0
    )

    # Get highest prediction scores

    top_indices = predictions[0].argsort()[-(top_k + 1):][::-1]

    recommendations = []

    for index in top_indices:

        # Exclude the selected title

        if index == content_id:
            continue

        # Content_ID matches the row position
        recommended_row = df.iloc[int(index)]

        recommendations.append({

            "content_id": int(recommended_row["Content_ID"]),

            "title": recommended_row["Title"],

            "language": recommended_row["Language Indicator"],

            "content_type": recommended_row["Content Type"],

            "hours_viewed": int(recommended_row["Hours Viewed"]),

            "prediction_score": float(predictions[0][index])

        })

        if len(recommendations) == top_k:
            break

    selected_title = {

        "content_id": content_id,

        "title": content_row["Title"],

        "language": content_row["Language Indicator"],

        "content_type": content_row["Content Type"]

    }

    return selected_title, recommendations


# ==========================================
# 5. HOME API
# ==========================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# ==========================================
# 6. HEALTH CHECK API
# ==========================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "status": "healthy",

        "model_loaded": True,

        "total_titles": len(df)

    })


# ==========================================
# 7. RECOMMENDATION API - GET
# ==========================================

@app.route("/recommend", methods=["GET"])
def get_recommendations():

    title = request.args.get("title", "")

    try:
        top_k = int(request.args.get("top_k", 5))
    except ValueError:
        return jsonify({
            "error": "top_k must be an integer"
        }), 400

    if not title.strip():
        return jsonify({
            "error": "Please provide a title"
        }), 400

    if not 1 <= top_k <= 20:
        return jsonify({
            "error": "top_k must be between 1 and 20"
        }), 400

    selected, recommendations = recommend_similar(
        title,
        top_k
    )

    if selected is None:
        return jsonify({
            "error": "Title not found"
        }), 404

    return jsonify({

        "selected_title": selected,

        "recommendations": recommendations,

        "total_recommendations": len(recommendations)

    })


# ==========================================
# 8. RECOMMENDATION API - POST
# ==========================================

@app.route("/recommend", methods=["POST"])
def post_recommendations():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please provide valid JSON"
        }), 400

    title = data.get("title", "")

    top_k = data.get("top_k", 5)

    if not isinstance(title, str) or not title.strip():
        return jsonify({
            "error": "Title is required"
        }), 400

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        return jsonify({
            "error": "top_k must be an integer"
        }), 400

    if not 1 <= top_k <= 20:
        return jsonify({
            "error": "top_k must be between 1 and 20"
        }), 400

    selected, recommendations = recommend_similar(
        title,
        top_k
    )

    if selected is None:
        return jsonify({
            "error": "Title not found"
        }), 404

    return jsonify({

        "selected_title": selected,

        "recommendations": recommendations,

        "total_recommendations": len(recommendations)

    })


# ==========================================
# 9. SEARCH TITLES API
# ==========================================

@app.route("/titles", methods=["GET"])
def search_titles():

    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({
            "error": "Please provide a search query"
        }), 400

    results = df[
        df["Title"].str.contains(
            re.escape(query),
            case=False,
            na=False
        )
    ].head(20)

    titles = results[
        [
            "Content_ID",
            "Title",
            "Language Indicator",
            "Content Type"
        ]
    ].to_dict(orient="records")

    return jsonify({

        "results": titles,

        "total_results": len(titles)

    })


# ==========================================
# 10. RUN FLASK
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
