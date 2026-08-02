# 🚀 Enterprise Telecom Churn Prediction Microservice

An end-to-end, production-grade Machine Learning pipeline and web application for predicting customer churn in the telecom industry. The system utilizes LightGBM, CatBoost, and Pseudo-Labeling ensembles served via a high-performance **FastAPI microservice** and an interactive **Streamlit dashboard**.

---

## 📌 Architecture Highlights

* **Decoupled Architecture:** Separates the frontend dashboard (`Streamlit`) from the core inference engine (`FastAPI`).
* **Deterministic Stateful Encoding:** Utilizes Scikit-Learn's `OrdinalEncoder` fit on training data and persisted to disk (`encoder.pkl`) to ensure zero feature drift between training and live REST inference.
* **Full K-Fold Ensembling:** Aggregates predictions across all 10 trained Stratified K-Fold binaries to prevent variance and preserve cross-validation integrity.
* **Strict Payload Validation:** Uses **Pydantic** schemas with custom `@model_validator` functions to enforce data types, range bounds, and domain consistency rules (e.g., catching impossible option combinations).
* **High-Throughput Serving:** Leverages lazy model loading and asynchronous thread execution (`asyncio.to_thread`) to process single and batch CSV inferences without blocking the API event loop.

---

## 📁 Repository Structure

```text
├── app.py              # Streamlit frontend dashboard (Single Profile & Batch CSV)
├── api.py              # FastAPI microservice serving REST prediction endpoints
├── schemas.py          # Pydantic data validation contracts & domain rules
├── utils.py            # Centralized pipeline logic, stateful encoding, & config
├── trainlgb.py         # Trains 10-fold LightGBM models
├── traincat.py         # Trains 10-fold CatBoost models
├── trainpsuedo.py      # Trains 10-fold LightGBM models with pseudo-labeled data
├── ensemble.py         # Evaluates blended out-of-fold predictions & test submission
├── requirements.txt    # Python package dependencies
└── outputs/            # Saved model artifacts (.pkl), encoder rules, & CSV outputs
