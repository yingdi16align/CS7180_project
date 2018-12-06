"""
To run this app, in your terminal:
> python iris_classification_api.py
"""
import connexion
from sklearn.externals import joblib

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='swagger/')
application = app.app



# Implement a simple health check function (GET)
def health():
    # Test to make sure our service is actually healthy
    try:
        predict(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
    except:
        return {"Message": "Service is unhealthy"}, 500

    return {"Message": "Service is OK"}

# Implement our predict function
def predict(model_select, case_submitted_day, case_submitted_month, case_submitted_year, soc_name, naics_code, total_workers, 
    full_time_position, prevailing_wage, pw_source, pw_source_year, wage_rate_of_pay_from, h1b_dependent, willful_violator, worksite_state):
    # Accept the feature values provided as part of our POST
    # Use these as input to clf.predict()
    if (model_select == 1):
        # Load our pre-trained model
        clf = joblib.load('./model/h1b_classifier_dt.joblib')
        prediction = clf.predict([[case_submitted_day, case_submitted_month, case_submitted_year, soc_name, naics_code, 
            total_workers, full_time_position, prevailing_wage, pw_source, pw_source_year, wage_rate_of_pay_from, h1b_dependent, willful_violator, worksite_state]])
    elif (model_select == 2):
        clf = joblib.load('./model/h1b_classifier_rf.joblib')
        prediction = clf.predict([[case_submitted_day, case_submitted_month, case_submitted_year, soc_name, naics_code, 
            total_workers, full_time_position, prevailing_wage, pw_source, pw_source_year, wage_rate_of_pay_from, h1b_dependent, willful_violator, worksite_state]])
    else:
        clf = joblib.load('./model/h1b_classifier_nb.joblib')
        prediction = clf.predict([[case_submitted_day, case_submitted_month, case_submitted_year, soc_name, naics_code, 
            total_workers, full_time_position, prevailing_wage, pw_source, pw_source_year, wage_rate_of_pay_from, h1b_dependent, willful_violator, worksite_state]])


    # Map the predicted value to an actual class
    if prediction[0] == 0:
        predicted_class = "Certified"
    else:
        predicted_class = "Denied"

    # Return the prediction as a json
    return {"prediction" : predicted_class}

# Read the API definition for our service from the yaml file
app.add_api("h1b_classification_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
