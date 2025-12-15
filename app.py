from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# =====================
# LOAD MODEL & SCALER
# =====================
model = pickle.load(open("kmeans_model_k4.pkl", "rb"))
scaler = pickle.load(open("scaler_model.pkl", "rb"))

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
                result = cluster_labels[cluster]["name"] + " - " + cluster_labels[cluster]["recommendation"]
        except Exception as e:
            error_msg = f"Ada kesalahan input: {str(e)}"

    return render_template("form.html", result=result, error=error_msg)

# =====================
# ROUTE UNTUK MOBILE/API
# =====================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.json
        product_name = data.get("product_name", "Unknown")
        total_qty = data["total_qty"]
        total_revenue = data["total_revenue"]
        total_transactions = data["total_transactions"]

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

# =====================
# RUN SERVER
# =====================
if __name__ == "__main__":
    app.run(debug=True)
