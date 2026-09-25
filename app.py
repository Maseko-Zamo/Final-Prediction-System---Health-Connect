from flask import Flask, render_template, request
import pandas as pd
import joblib
import csv
import os
from datetime import datetime


app = Flask(
    __name__,
    template_folder=".",
    static_folder="static"
)


# -----------------------------------
# LOAD TRAINED MODEL
# -----------------------------------

model = joblib.load(
    "health_connect_logistic_regression_model.pkl"
)


# -----------------------------------
# CSV FILE
# -----------------------------------

CSV_FILE = "prediction_history.csv"


# -----------------------------------
# SAVE PREDICTION
# -----------------------------------

def save_prediction(data):

    file_exists = os.path.isfile(CSV_FILE)

    with open(
        CSV_FILE,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "Timestamp",
            "Booking Lead Days",
            "Age",
            "Previous No Shows",
            "Distance to Clinic (km)",
            "Waiting Time (minutes)",
            "Booking Days Category",
            "Distance Category",
            "Reminder Received",
            "Predicted Outcome",
            "Probability (%)"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


# -----------------------------------
# HOME
# -----------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None

    if request.method == "POST":

        # -----------------------------------
        # GET USER INPUTS
        # -----------------------------------

        booking_lead_days = float(
            request.form.get("booking_lead_days")
        )

        age = float(
            request.form.get("age")
        )

        previous_no_shows = float(
            request.form.get("previous_no_shows")
        )

        distance_to_clinic = float(
            request.form.get("distance_to_clinic")
        )

        waiting_time = float(
            request.form.get("waiting_time")
        )

        booking_days_category = request.form.get(
            "booking_days_category"
        )

        distance_category = request.form.get(
            "distance_category"
        )

        reminder_received = request.form.get(
            "reminder_received"
        )


        # -----------------------------------
        # ENCODING
        # -----------------------------------

        booking_days_encoding = {
            "Short": 0,
            "Medium": 1,
            "Long": 2,
            "Very Long": 3
        }

        distance_encoding = {
            "Near": 0,
            "Moderate": 1,
            "Far": 2,
            "Very Far": 3
        }

        reminder_encoding = {
            "Yes": 1,
            "No": 0
        }


        booking_days_encoded = booking_days_encoding[
            booking_days_category
        ]

        distance_encoded = distance_encoding[
            distance_category
        ]

        reminder_encoded = reminder_encoding[
            reminder_received
        ]


        # -----------------------------------
        # CREATE MODEL INPUT
        # -----------------------------------

        user_data = pd.DataFrame({
            "booking_lead_days": [
                booking_lead_days
            ],

            "age": [
                age
            ],

            "previous_no_shows": [
                previous_no_shows
            ],

            "distance_to_clinic_km": [
                distance_to_clinic
            ],

            "waiting_time_minutes": [
                waiting_time
            ],

            "booking_days_category": [
                booking_days_encoded
            ],

            "Distance_Category": [
                distance_encoded
            ],

            "reminder_received": [
                reminder_encoded
            ]
        })


        # -----------------------------------
        # PREDICTION
        # -----------------------------------

        prediction_result = model.predict(
            user_data
        )[0]


        # -----------------------------------
        # OUTCOME
        # -----------------------------------

        if prediction_result == 1:

            prediction = "Likely to Attend"

        else:

            prediction = "Likely to Not Show Up"


        # -----------------------------------
        # PROBABILITY
        # -----------------------------------

        probabilities = model.predict_proba(
            user_data
        )[0]

        predicted_index = list(
            model.classes_
        ).index(
            prediction_result
        )

        probability = round(
            probabilities[predicted_index] * 100,
            2
        )


        # -----------------------------------
        # SAVE EVERYTHING TO CSV
        # -----------------------------------

        prediction_record = {
            "Timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "Booking Lead Days": booking_lead_days,

            "Age": age,

            "Previous No Shows": previous_no_shows,

            "Distance to Clinic (km)": distance_to_clinic,

            "Waiting Time (minutes)": waiting_time,

            "Booking Days Category": booking_days_category,

            "Distance Category": distance_category,

            "Reminder Received": reminder_received,

            "Predicted Outcome": prediction,

            "Probability (%)": probability
        }

        save_prediction(prediction_record)


    # -----------------------------------
    # DISPLAY WEBSITE
    # -----------------------------------

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )


# -----------------------------------
# RUN
# -----------------------------------

if __name__ == "__main__":
    app.run(debug=True)