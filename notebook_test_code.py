import pandas as pd
import pickle

# Load the model
with open("subscription_model.pkl", "rb") as f:
    model = pickle.load(f)

# Create a test input matching training features
test_input = pd.DataFrame([{
    "Age": 30,
    "Gender": 1,
    "Item Purchased": 0,
    "Category": 0,
    "Location": 0,
    "Size": 1,
    "Color": 2,
    "Season": 1,
    "Review Rating": 4.5,
    "Shipping Type": 0,
    "Discount Applied": 1,
    "Promo Code Used": 1,
    "Previous Purchases": 5,
    "Payment Method": 0,
    "Frequency of Purchases": 10
}])

# Predict
prediction = model.predict(test_input)[0]
print("Prediction:", "Subscribed" if prediction == 1 else "Not Subscribed")
