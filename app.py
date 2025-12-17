from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

# Gunakan _name_ (double underscore)
app = Flask(_name_)

# =====================
# LOAD MODEL & SCALER
# =====================
# Tambahkan error handling agar kita tahu kalau file hilang di Vercel
try:
    model = pickle.load(open("kmeans_model_k4.pkl", "rb"))
    scaler = pickle.load(open("scaler_model.pkl", "rb"))
except FileNotFoundError:
    print("ERROR: Model file not found. Pastikan file .pkl sudah di-upload.")
    model = None
    scaler = None

# =====================
# CLUSTER KE KATEGORI
# =====================
cluster_labels = {
    0: {"name": "Star", "recommendation": "Jaga stok dan prioritaskan display"},
    1: {"name": "Premium", "recommendation": "Stok stabil dan pantau tren"},
    2: {"name": "Economy", "recommendation": "Optimalkan harga dan promosi"},
    3: {"name": "Slow", "recommendation": "Kurangi stok atau bundling promo"}
}

# =====================
# ROUTE UNTUK WEB FORM
# =====================
@app.route("/", methods=["GET", "POST"])
def index():
    if model is None or scaler is None:
        return "Model tidak ditemukan di server. Cek log deployment.", 500

    result = None
    error_msg = None
    
    if request.method == "POST":
        try:
            total_qty = float(request.form.get("total_qty", 0))
            total_revenue = float(request.form.get("total_revenue", 0))
            total_transactions = float(request.form.get("total_transactions", 0))

            if total_qty <= 0 or total_transactions <= 0:
                error_msg = "Total Qty dan Total Transaksi harus lebih dari 0."
            else:
                # Feature engineering
                price_mean = total_revenue / total_qty
                velocity = total_qty / total_transactions
                revenue_per_transaction = total_revenue / total_transactions

                X = np.array([[total_qty, price_mean, velocity, revenue_per_transaction]])
                X_scaled = scaler.transform(X)
                cluster = int(model.predict(X_scaled)[0])
                result = f"{cluster_labels[cluster]['name']} - {cluster_labels[cluster]['recommendation']}"
        except Exception as e:
            error_msg = f"Ada kesalahan input: {str(e)}"

    return render_template("form.html", result=result, error=error_msg)

# =====================
# ROUTE UNTUK MOBILE/API
# =====================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json
        product_name = data.get("product_name", "Unknown")
        total_qty = float(data["total_qty"])
        total_revenue = float(data["total_revenue"])
        total_transactions = float(data["total_transactions"])

        # Feature engineering
        price_mean = total_revenue / total_qty
        velocity = total_qty / total_transactions
        revenue_per_transaction = total_revenue / total_transactions

        X = np.array([[total_qty, price_mean, velocity, revenue_per_transaction]])
        X_scaled = scaler.transform(X)
        cluster = int(model.predict(X_scaled)[0])

        response = {
            "product_name": product_name,
            "category_id": cluster,
            "category_name": cluster_labels[cluster]["name"],
            "recommendation": cluster_labels[cluster]["recommendation"]
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Opsional di Vercel, tapi berguna untuk tes lokal
if _name_ == "_main_":
    app.run(debug=True)