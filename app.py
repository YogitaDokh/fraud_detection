from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the trained KNN model
MODEL_PATH = 'KNN_model.pkl'
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
else:
    model = None

# Single-file HTML template with modern UI styling using Bootstrap 5
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fraud Guard AI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f4f6f9; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .hero-banner {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 40px 20px;
            border-radius: 0 0 25px 25px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .card { border: none; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
        .form-label { font-weight: 600; color: #495057; font-size: 0.9rem; }
        .form-control, .form-select { border-radius: 8px; padding: 10px; border: 1px solid #ced4da; }
        .form-control:focus, .form-select:focus { box-shadow: 0 0 0 0.25rem rgba(42, 82, 152, 0.25); border-color: #2a5298; }
        .btn-primary { background-color: #2a5298; border: none; padding: 12px; font-weight: 600; border-radius: 8px; transition: all 0.3s; }
        .btn-primary:hover { background-color: #1e3c72; transform: translateY(-1px); }
        .result-box { border-radius: 12px; padding: 25px; text-align: center; font-size: 1.25rem; font-weight: bold; margin-top: 20px; }
        .result-safe { background-color: #d1e7dd; color: #0f5132; border: 2px solid #badbcc; }
        .result-fraud { background-color: #f8d7da; color: #842029; border: 2px solid #f5c2c7; }
    </style>
</head>
<body>

    <div class="hero-banner text-center">
        <h1 class="display-5 fw-bold">🛡️ Fraud Guard AI</h1>
        <p class="lead opacity-75">Real-time Transaction Fraud Risk Assessment Engine</p>
    </div>

    <div class="container mb-5">
        {% if model_error %}
        <div class="alert alert-danger text-center" role="alert">
            <strong>Error:</strong> {{ model_error }}
        </div>
        {% endif %}

        <div class="row g-4">
            <div class="col-lg-8">
                <div class="card p-4">
                    <h4 class="mb-4 text-primary">📥 Transaction Details</h4>
                    <form method="POST" action="/">
                        <div class="row g-3">
                            <div class="col-md-6">
                                <label class="form-label">Transaction Amount ($)</label>
                                <input type="number" step="0.01" class="form-control" name="transaction_amount" value="{{ inputs.transaction_amount or 150.0 }}" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Hour of Day (0-23)</label>
                                <input type="number" min="0" max="23" class="form-control" name="hour_of_day" value="{{ inputs.hour_of_day or 14 }}" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Is Weekend?</label>
                                <select class="form-select" name="is_weekend">
                                    <option value="0" {% if inputs.is_weekend == '0' %}selected{% endif %}>No</option>
                                    <option value="1" {% if inputs.is_weekend == '1' %}selected{% endif %}>Yes</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Number of Items</label>
                                <input type="number" min="1" class="form-control" name="num_items" value="{{ inputs.num_items or 2 }}" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Customer Age</label>
                                <input type="number" min="18" class="form-control" name="customer_age" value="{{ inputs.customer_age or 35 }}" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Previous Transactions Count</label>
                                <input type="number" min="0" class="form-control" name="prev_transactions" value="{{ inputs.prev_transactions or 5 }}" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Distance from Home (miles)</label>
                                <input type="number" step="0.1" class="form-control" name="distance_from_home" value="{{ inputs.distance_from_home or 12.5 }}" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Device Type (ID Encoded)</label>
                                <input type="number" class="form-control" name="device_type" value="{{ inputs.device_type or 1 }}" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Network Quality Index</label>
                                <input type="number" class="form-control" name="network_quality" value="{{ inputs.network_quality or 3 }}" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Is First Transaction?</label>
                                <select class="form-select" name="is_first_transaction">
                                    <option value="0" {% if inputs.is_first_transaction == '0' %}selected{% endif %}>No</option>
                                    <option value="1" {% if inputs.is_first_transaction == '1' %}selected{% endif %}>Yes</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Store Type (ID Encoded)</label>
                                <input type="number" class="form-control" name="store_type" value="{{ inputs.store_type or 2 }}" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Velocity Score</label>
                                <input type="number" step="0.1" class="form-control" name="velocity_score" value="{{ inputs.velocity_score or 1.2 }}" required>
                            </div>
                        </div>
                        <div class="mt-4">
                            <button type="submit" class="btn btn-primary w-100">🚀 Analyze Transaction</button>
                        </div>
                    </form>
                </div>
            </div>

            <div class="col-lg-4">
                <div class="card p-4 h-100 d-flex flex-column justify-content-start">
                    <h4 class="mb-4 text-primary">🔍 Evaluation Result</h4>
                    
                    {% if prediction is not none %}
                        {% if prediction == 1 %}
                            <div class="result-box result-fraud">
                                🚨 HIGH RISK DETECTED
                                <div class="fs-6 fw-normal mt-2">This transaction aligns closely with signature fraudulent patterns.</div>
                            </div>
                        {% else %}
                            <div class="result-box result-safe">
                                ✅ TRANSACTION APPROVED
                                <div class="fs-6 fw-normal mt-2">Low risk evaluation. Safe to proceed with settlement.</div>
                            </div>
                        {% endif %}

                        {% if probability is not none %}
                            <div class="mt-4 text-center">
                                <p class="text-muted mb-1">Model Confidence Probability</p>
                                <h3 class="fw-bold text-dark">{{ "%.2f"|format(probability) }}%</h3>
                            </div>
                        {% endif %}
                    {% else %}
                        <div class="text-center text-muted my-auto py-5">
                            <p class="mb-0">Fill out the form details and hit <strong>Analyze Transaction</strong> to trigger the detection model.</p>
                        </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None
    model_error = None
    inputs = {}

    if model is None:
        model_error = "KNN_model.pkl not found in root directory. Please upload your model artifact."

    if request.method == 'POST':
        # Safely grab inputs from the user form submission
        inputs = {
            'transaction_amount': float(request.form.get('transaction_amount', 0)),
            'hour_of_day': int(request.form.get('hour_of_day', 0)),
            'is_weekend': int(request.form.get('is_weekend', 0)),
            'num_items': int(request.form.get('num_items', 0)),
            'customer_age': int(request.form.get('customer_age', 0)),
            'prev_transactions': int(request.form.get('prev_transactions', 0)),
            'distance_from_home': float(request.form.get('distance_from_home', 0)),
            'device_type': int(request.form.get('device_type', 0)),
            'network_quality': int(request.form.get('network_quality', 0)),
            'is_first_transaction': int(request.form.get('is_first_transaction', 0)),
            'store_type': int(request.form.get('store_type', 0)),
            'velocity_score': float(request.form.get('velocity_score', 0))
        }

        if model is not None:
            try:
                # Structure features exactly matching the trained model pipeline
                input_df = pd.DataFrame([inputs])
                
                # Fetch base prediction class
                prediction = int(model.predict(input_df)[0])
                
                # Try fetching structural probabilities if the KNN supports it
                if hasattr(model, "predict_proba"):
                    prob_array = model.predict_proba(input_df)[0]
                    # Show target confidence depending on output target value
                    probability = float(prob_array[1] if prediction == 1 else prob_array[0]) * 100
            except Exception as e:
                model_error = f"Prediction failed structure match: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE, 
        prediction=prediction, 
        probability=probability, 
        model_error=model_error, 
        inputs=inputs
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
