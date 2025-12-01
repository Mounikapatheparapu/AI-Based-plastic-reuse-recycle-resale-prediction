# app.py  -- final troubleshooting version
import os
import importlib
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for

# compatibility shim for sklearn pickles
try:
    _mod = importlib.import_module("sklearn.compose._column_transformer")
    if not hasattr(_mod, "_RemainderColsList"):
        class _RemainderColsList(list): pass
        setattr(_mod, "_RemainderColsList", _RemainderColsList)
except Exception:
    pass

app = Flask(__name__, template_folder="templates", static_folder="static")

# --- config ---
MODELS_DIR = "models"
PREPROCESSOR_PKL = os.path.join(MODELS_DIR, "preprocessor.pkl")
MODEL_FILES = {
    "Resale_Value": os.path.join(MODELS_DIR, "Resale_Value_GradientBoosting_Model.pkl"),
    "Recycle_Value": os.path.join(MODELS_DIR, "Recycle_Value_GradientBoosting_Model.pkl"),
    "Reuse_Score": os.path.join(MODELS_DIR, "Reuse_Score_GradientBoosting_Model.pkl")
}

# Load models (graceful)
preprocessor = None
models = {}
models_loaded = False
def try_load_models():
    global preprocessor, models, models_loaded
    try:
        if os.path.isfile(PREPROCESSOR_PKL):
            preprocessor = joblib.load(PREPROCESSOR_PKL)
        else:
            raise FileNotFoundError(PREPROCESSOR_PKL)
        for k, p in MODEL_FILES.items():
            if os.path.isfile(p):
                models[k] = joblib.load(p)
            else:
                raise FileNotFoundError(p)
        models_loaded = True
        print("Models loaded.")
    except Exception as e:
        models_loaded = False
        print("Models NOT loaded:", e)

try_load_models()

condition_mapping = {'cracked': 1, 'used': 2, 'good': 3, 'new': 4}

def build_input_df(payload):
    return pd.DataFrame([{
        "Plastic_Type": payload.get("ptype"),
        "Condition": payload.get("condition"),
        "Weight_Kg": payload.get("weight"),
        "Age_Months": payload.get("age"),
        "Original_Price": payload.get("price"),
        "Location": payload.get("location")
    }])

def predict_all_targets(df_raw):
    if models_loaded and preprocessor is not None and models:
        X = df_raw.copy()
        if "Condition" in X.columns:
            def _map_cond(v):
                if pd.isna(v): return 0
                return condition_mapping.get(str(v).lower().strip(), 0)
            X["Condition"] = X["Condition"].apply(_map_cond)
        transformed = preprocessor.transform(X)
        try:
            cols = preprocessor.get_feature_names_out()
        except Exception:
            cols = [f"f{i}" for i in range(transformed.shape[1])]
        X_enc = pd.DataFrame(transformed, columns=cols)
        out = {}
        for k, m in models.items():
            out[k] = float(m.predict(X_enc)[0])
        return out
    else:
        # placeholder fallback
        r = df_raw.iloc[0]
        price = float(r.get("Original_Price") or 0)
        weight = float(r.get("Weight_Kg") or 0)
        age = float(r.get("Age_Months") or 0)
        return {
            "Resale_Value": round(max(0, price * 0.2 - age * 0.1), 2),
            "Recycle_Value": round(max(0, weight * 12.0), 2),
            "Reuse_Score": round(max(0, 100 - age), 2)
        }

# --- helper: render template or raw file if jinja fails ---
def render_or_static(fname):
    tpl = os.path.join(app.template_folder or "templates", fname)
    if os.path.isfile(tpl):
        try:
            return render_template(fname)
        except Exception as e:
            print(f"render_template failed for {fname}: {e}; sending raw file.")
            return send_from_directory(app.template_folder, fname)
    else:
        return "Template not found on server", 404

# --- pages ---
@app.route("/")
def index():
    return render_or_static("index.html")

@app.route("/ourai")
def ourai():
    return render_or_static("ourai.html")

@app.route("/about")
def about():
    return render_or_static("about.html")

@app.route("/contact")
def contact():
    return render_or_static("contact.html")

# -- routes debug page: lists url_map so you can click from browser
@app.route("/routes")
def routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(str(rule))
    # simple clickable page
    links = "".join([f'<li><a href="{r}">{r}</a></li>' for r in routes])
    return f"<h3>Registered routes</h3><ul>{links}</ul><p>Open / or /ourai.</p>"

# --- prediction API ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error":"Invalid JSON"}), 400
    try:
        df = build_input_df(payload)
        preds = predict_all_targets(df)
    except Exception as e:
        return jsonify({"error":"Prediction failed","details":str(e)}), 500
    return jsonify({
        "resale_value": preds.get("Resale_Value"),
        "recycle_score": preds.get("Recycle_Value"),
        "reuse_score": preds.get("Reuse_Score"),
        "models_loaded": models_loaded
    })

# --- catch-all: if user visits unknown path, return index (prevents "Not Found" for mistyped URLs) ---
@app.route("/<path:subpath>")
def catch_all(subpath):
    # If file exists in templates and matches subpath, serve it; else serve index.html
    candidate = f"{subpath}"
    tpl_path = os.path.join(app.template_folder or "templates", candidate)
    if os.path.isfile(tpl_path) and candidate.endswith(".html"):
        return render_or_static(candidate)
    # serve index so clicking unknown path doesn't show flask's plain 404 page
    return render_or_static("index.html")

# --- startup diagnostics ---
def startup_info():
    print("\n--- STARTUP INFO ---")
    print("Working dir:", os.path.abspath("."))
    print("Templates folder exists:", os.path.isdir(app.template_folder or "templates"))
    print("Templates found:", [f for f in os.listdir(app.template_folder or "templates") if f.endswith(".html")] if os.path.isdir(app.template_folder or "templates") else [])
    print("Models folder exists:", os.path.isdir(MODELS_DIR))
    print("preprocessor exists:", os.path.isfile(PREPROCESSOR_PKL))
    for k, p in MODEL_FILES.items():
        print(f"{k} exists:", os.path.isfile(p))
    print("Registered routes:")
    for r in app.url_map.iter_rules():
        print(" ", r)
    print("--- END STARTUP INFO ---\n")

if __name__ == "__main__":
    startup_info()
    # When running locally, use PORT env var if present (Railway supplies PORT)
    port = int(os.environ.get("PORT", 8080))
    # Bind to all interfaces so host can reach container
    app.run(host="0.0.0.0", port=port, debug=False)
