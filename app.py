from flask import Flask, request, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained KNN model
MODEL_PATH = "KNN_model.pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Expected order of feature names from the model metadata
FEATURE_NAMES = [
    "transaction_amount", "hour_of_day", "is_weekend", "num_items", 
    "customer_age", "prev_transactions", "distance_from_home", "device_type", 
    "network_quality", "is_first_transaction", "store_type", "velocity_score"
]

# Attractive HTML Template for the Web UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KNN Prediction Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #f4f6f9;
            --card-bg: #ffffff;
            --text-color: #2d3748;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --border: #e2e8f0;
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }
        .container {
            max-width: 700px;
            width: 100%;
            background: var(--card-bg);
            padding: 35px;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        }
        h1 {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #1a202c;
        }
        p.subtitle {
            color: #718096;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }
        @media (max-width: 600px) {
            .grid { grid-template-columns: 1fr; }
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 6px;
            text-transform: capitalize;
            color: #4a5568;
        }
        input, select {
            padding: 10px 14px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            background-color: #f8fafc;
            transition: all 0.2s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            background-color: #fff;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }
        button {
            grid-column: span 2;
            background-color: var(--primary);
            color: white;
            padding: 14px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
            margin-top: 10px;
        }
        @media (max-width: 600px) { button { grid-column: span 1; } }
        button:hover { background-color: var(--primary-hover); }
        .result-box {
            margin-top: 30px;
            padding: 20px;
            border-radius: 8px;
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            display: none;
            text-align: center;
        }
        .result-box.error {
            background-color: #fef2f2;
            border-color: #fecaca;
            color: #991b1b;
        }
        .result-title {
            font-weight: 600;
            font-size: 18px;
            color: #166534;
            margin-bottom: 4px;
        }
        .result-val { font-size: 15px; color: #1e293b; }
    </style>
</head>
<body>

<div class="container">
    <h1>KNN Model Predictor</h1>
    <p class="subtitle">Fill out the features below to retrieve a real-time intelligence prediction.</p>
    
    <form id="predictionForm" class="grid">
        <div class="form-group">
            <label>Transaction Amount</label>
            <input type="number" step="any" name="transaction_amount" value="150.00" required>
        </div>
        <div class="form-group">
            <label>Hour of Day (0-23)</label>
            <input type="number" min="0" max="23" name="hour_of_day" value="14" required>
        </div>
        <div class="form-group">
            <label>Is Weekend</label>
            <select name="is_weekend">
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select>
        </div>
        <div class="form-group">
            <label>Num Items</label>
            <input type="number" min="1" name="num_items" value="2" required>
        </div>
        <div class="form-group">
            <label>Customer Age</label>
            <input type="number" min="0" name="customer_age" value="30" required>
        </div>
        <div class="form-group">
            <label>Prev Transactions</label>
            <input type="number" min="0" name="prev_transactions" value="4" required>
        </div>
        <div class="form-group">
            <label>Distance From Home</label>
            <input type="number" step="any" name="distance_from_home" value="3.5" required>
        </div>
        <div class="form-group">
            <label>Device Type</label>
            <input type="number" name="device_type" value="1" required>
        </div>
        <div class="form-group">
            <label>Network Quality</label>
            <input type="number" name="network_quality" value="3" required>
        </div>
        <div class="form-group">
            <label>Is First Transaction</label>
            <select name="is_first_transaction">
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select>
        </div>
        <div class="form-group">
            <label>Store Type</label>
            <input type="number" name="store_type" value="2" required>
        </div>
        <div class="form-group">
            <label>Velocity Score</label>
            <input type="number" step="any" name="velocity_score" value="0.75" required>
        </div>
        
        <button type="submit">Generate Prediction</button>
    </form>

    <div id="resultBox" class="result-box">
        <div class="result-title" id="resTitle">Prediction Generated</div>
        <div class="result-val" id="resContent"></div>
    </div>
</div>

<script>
    document.getElementById('predictionForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Convert form entries directly into URL parameters for a standard form POST
        const formData = new FormData(e.target);
        const searchParams = new URLSearchParams(formData);

        const resultBox = document.getElementById('resultBox');
        const resTitle = document.getElementById('resTitle');
        const resContent = document.getElementById('resContent');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: searchParams.toString()
            });
            
            // Read response content as plain text string split by pipe delimiter
            const textResponse = await response.text();

            resultBox.style.display = 'block';
            if (response.ok) {
                resultBox.className = "result-box";
                resTitle.innerText = "Prediction Success!";
                resTitle.style.color = "#166534";
                
                // Parse custom clean text response format (e.g., "prediction,confidence")
                const parts = textResponse.split('|');
                let displayHTML = `<strong>Prediction Class:</strong> ${parts[0]}`;
                if(parts.length > 1 && parts[1] !== "None") {
                    let confidencePercent = (parseFloat(parts[1]) * 100).toFixed(2);
                    displayHTML += `<br><strong>Confidence:</strong> ${confidencePercent}%`;
                }
                resContent.innerHTML = displayHTML;
            } else {
                throw new Error(textResponse || "Unknown server error");
            }
        } catch (err) {
            resultBox.style.display = 'block';
            resultBox.className = "result-box error";
            resTitle.innerText = "Error Occurred";
            resTitle.style.color = "#991b1b";
            resContent.innerText = err.message;
        }
    });
</script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect features using standard form parameters
        features = []
        missing_features = []
        
        for feature in FEATURE_NAMES:
            value = request.form.get(feature)
            if value is not None:
                features.append(float(value))
            else:
                missing_features.append(feature)

        if missing_features:
            return f"Missing required features: {', '.join(missing_features)}", 400

        input_data = np.array([features])
        prediction = model.predict(input_data)
        
        try:
            probabilities = model.predict_proba(input_data)
            confidence = float(np.max(probabilities))
        except AttributeError:
            confidence = "None"

        # Return a simple pipe-separated plain text string instead of JSON
        return f"{int(prediction[0])}|{confidence}", 200

    except Exception as e:
        return f"Server Error: {str(e)}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
