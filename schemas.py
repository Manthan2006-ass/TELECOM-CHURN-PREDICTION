from pydantic import BaseModel, Field, model_validator
from typing import Literal, List, Dict, Any, Optional

class RiskFactor(BaseModel):
    feature: str
    impact: Literal["HIGH RISK", "MODERATE RISK", "PROTECTIVE / SAFE"]
    detail: str

class CustomerProfileSchema(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No phone service", "No", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "Yes", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["No", "Yes", "No internet service"]
    TechSupport: Literal["No", "Yes", "No internet service"]
    StreamingTV: Literal["No", "Yes", "No internet service"]
    StreamingMovies: Literal["No", "Yes", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(..., ge=0.0)
    TotalCharges: float = Field(..., ge=0.0)

    # Domain Inconsistency Validator for Impossible Option Combinations
    @model_validator(mode="after")
    def validate_domain_rules(self):
        if self.PhoneService == "No" and self.MultipleLines in ["Yes", "No"]:
            raise ValueError("MultipleLines must be 'No phone service' when PhoneService is 'No'.")
            
        internet_features = [
            self.OnlineSecurity, self.OnlineBackup, self.DeviceProtection,
            self.TechSupport, self.StreamingTV, self.StreamingMovies
        ]
        
        if self.InternetService == "No":
            if any(feat in ["Yes", "No"] for feat in internet_features):
                raise ValueError("Internet add-on features must be 'No internet service' when InternetService is 'No'.")
        else:
            if any(feat == "No internet service" for feat in internet_features):
                raise ValueError("Internet add-on features cannot be 'No internet service' when InternetService is active.")
                
        return self

class PredictionOutputSchema(BaseModel):
    churn_probability: float
    risk_status: str
    risk_tier: str
    decision_threshold: float
    recommendation: str
    key_risk_factors: List[RiskFactor] = []
