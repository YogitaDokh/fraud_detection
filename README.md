# 🛡️ Fraud Guard AI - Risk Assessment Engine

Fraud Guard AI is a production-ready Flask web application that wraps a pre-trained K-Nearest Neighbors (KNN) machine learning classification model to evaluate transaction security risks in real time. 

Built using a responsive single-file layout, it features a clean Bootstrap 5 dashboard designed for desktop and mobile devices, making model inference straightforward and interactive.

---

## 🔗 Live Demo
You can access the live, fully interactive application here:  
👉 **[Launch Fraud Guard AI Dashboard](https://fraud-detection-7ti2.onrender.com/)** 

---

## 🚀 Key App Capabilities

- **Single-File Framework:** Eliminates directory complexity by coupling the structural template layer and routing logic natively inside a single `app.py` script.
- **Dynamic Context Rendering:** Employs Jinja2 native server templating via `render_template_string` to dynamically reflect predictions, metrics, states, and data entries instantly upon evaluation.
- **Comprehensive Error Boundaries:** Built-in safeguards check for model persistence states (`KNN_models.pkl`) and structural pipeline validation layout mismatches dynamically.
- **Robust Feature Parsing:** Automatically captures standard form submissions, parses input types gracefully into memory arrays, and evaluates matching feature contexts.

---

## 🛠️ Model Feature Specifications

The underlying predictive pipeline expects exactly **12 structured inputs** fed in sequence into a pandas DataFrame layout:

1. **Transaction Amount ($):** Continuous numeric value representing the financial volume of the transaction.
2. **Hour of Day (0-23):** Integer indicating the exact time block of the day the activity originated.
3. **Is Weekend?:** Categorical binary code (`0` for weekdays, `1` for weekend intervals).
4. **Number of Items:** Total count of items bundled within the settlement basket.
5. **Customer Age:** Demographic age metric of the customer account holder.
6. **Previous Transactions Count:** Historical record volume logged by the client profile.
7. **Distance from Home (miles):** Spatial delta tracking physical transaction origin against local profile settings.
8. **Device Type (ID Encoded):** Discrete identifier tracking user hardware footprint classifications.
9. **Network Quality Index:** Integer evaluating latency/cellular performance attributes at submission time.
10. **Is First Transaction?:** Categorical binary flag signaling whether the transaction acts as a baseline lifecycle initialization event.
11. **Store Type (ID Encoded):** Discretely classified industry category marker matching merchant profiles.
12. **Velocity Score:** Quantitative frequency indicator monitoring rapid transaction sequences.

---

## 📦 Workspace Architecture

To execute the application seamlessly, configure your project root to match this exact layout:

```text
├── app.py                  # Single-file Flask web routing and dashboard code
├── requirements.txt        # Backend dependencies and server library tracking
└── KNN_models.pkl          # Serialized pre-trained KNN classifier asset
