from flask import Flask, render_template, request
import pickle
import pandas as pd


app = Flask(__name__)


# ---------------------------------------------------
# Load Models
# ---------------------------------------------------

with open(
    "decision_tree_regressor.pkl",
    "rb"
) as file:

    decision_tree = pickle.load(file)


with open(
    "random_forest_regressor.pkl",
    "rb"
) as file:

    random_forest = pickle.load(file)


with open(
    "regression_selector.pkl",
    "rb"
) as file:

    selector = pickle.load(file)


with open(
    "regression_feature_names.pkl",
    "rb"
) as file:

    feature_names = pickle.load(file)


with open(
    "regression_selected_features.pkl",
    "rb"
) as file:

    selected_features = pickle.load(file)


# ---------------------------------------------------
# Home Route
# ---------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    if request.method == "POST":

        try:

            # ----------------------------------------
            # Read Selected Feature Values
            # ----------------------------------------

            user_values = {}

            for feature in selected_features:

                user_values[feature] = float(
                    request.form[feature]
                )


            # ----------------------------------------
            # Create Full Feature DataFrame
            # ----------------------------------------

            complete_values = {}

            for feature in feature_names:

                if feature in user_values:

                    complete_values[feature] = (
                        user_values[feature]
                    )

                else:

                    complete_values[feature] = 0.0


            input_df = pd.DataFrame(
                [complete_values],
                columns=feature_names
            )


            # ----------------------------------------
            # Apply Same Feature Selection
            # ----------------------------------------

            input_selected = selector.transform(
                input_df
            )


            # ----------------------------------------
            # Decision Tree Prediction
            # ----------------------------------------

            dt_prediction = decision_tree.predict(
                input_selected
            )[0]


            # ----------------------------------------
            # Random Forest Prediction
            # ----------------------------------------

            rf_prediction = random_forest.predict(
                input_selected
            )[0]


            # ----------------------------------------
            # Results
            # ----------------------------------------

            result = {

                "decision_tree":
                    round(dt_prediction, 2),

                "random_forest":
                    round(rf_prediction, 2)

            }


        except Exception as e:

            error = (
                "Please enter valid numerical values."
            )

            print(e)


    return render_template(

        "index.html",

        result=result,

        error=error,

        features=selected_features

    )


# ---------------------------------------------------
# Run Flask
# ---------------------------------------------------

if __name__ == "__main__":

    app.run(debug=True)