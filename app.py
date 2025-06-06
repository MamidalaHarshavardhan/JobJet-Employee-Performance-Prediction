from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Department mappings to match HTML form exactly
department_map = {
    'sales': 0,
    'marketing': 1,
    'hr': 2,
    'human resources': 2,
    'it': 3, 
    'information technology': 3,
    'finance': 4,
    'operations': 5,
    'r&d': 6,
    'research & development': 6,
    'research and development': 6
}

# Gender mappings to match HTML form exactly
gender_map = {
    'male': 0,
    'female': 1,
    'other': 2
}

# Job role mappings - more comprehensive to handle various inputs
role_map = {
    # Basic roles
    'manager': 0,
    'analyst': 1,
    'developer': 2,
    'executive': 3,
    'associate': 4,
    'intern': 5,
    'consultant': 6,
    
    # Combined roles (common patterns)
    'sales manager': 0,
    'marketing manager': 0,
    'project manager': 0,
    'hr manager': 0,
    'finance manager': 0,
    'operations manager': 0,
    'senior manager': 0,
    
    'business analyst': 1,
    'data analyst': 1,
    'financial analyst': 1,
    'systems analyst': 1,
    'marketing analyst': 1,
    'senior analyst': 1,
    
    'software developer': 2,
    'web developer': 2,
    'full stack developer': 2,
    'frontend developer': 2,
    'backend developer': 2,
    'senior developer': 2,
    
    'sales executive': 3,
    'marketing executive': 3,
    'business executive': 3,
    'account executive': 3,
    'senior executive': 3,
    
    'sales associate': 4,
    'marketing associate': 4,
    'hr associate': 4,
    'finance associate': 4,
    'senior associate': 4,
    
    'marketing intern': 5,
    'sales intern': 5,
    'finance intern': 5,
    'hr intern': 5,
    'it intern': 5,
    
    'business consultant': 6,
    'management consultant': 6,
    'it consultant': 6,
    'hr consultant': 6,
    'senior consultant': 6,
    
    # Additional common roles
    'coordinator': 4,
    'specialist': 1,
    'representative': 4,
    'officer': 4,
    'supervisor': 0,
    'lead': 0,
    'senior': 0,
    'junior': 5
}

def get_best_match(input_text, mapping_dict):
    """Find the best match for input text in mapping dictionary"""
    input_lower = input_text.lower().strip()
    
    # Exact match first
    if input_lower in mapping_dict:
        return mapping_dict[input_lower]
    
    # Partial match - check if any key contains the input or vice versa
    for key in mapping_dict:
        if key in input_lower or input_lower in key:
            return mapping_dict[key]
    
    # Check individual words for job roles
    if mapping_dict == role_map:
        words = input_lower.split()
        for word in words:
            if word in mapping_dict:
                return mapping_dict[word]
    
    return -1

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        education = float(request.form['education'])
        age = float(request.form['age'])

        # Get raw values from form
        department_raw = request.form['department'].strip()
        job_role_raw = request.form['job_role'].strip()
        gender_raw = request.form['gender'].strip()
        
        # Debug: Print what we received
        print(f"Raw inputs - Department: '{department_raw}', Job Role: '{job_role_raw}', Gender: '{gender_raw}'")
        
        # Use smart matching
        department = get_best_match(department_raw, department_map)
        job_role = get_best_match(job_role_raw, role_map)
        gender = get_best_match(gender_raw, gender_map)
        
        # Debug: Print mapped values
        print(f"Mapped values - Department: {department}, Job Role: {job_role}, Gender: {gender}")

        experience = float(request.form['experience'])
        training_score = float(request.form['training_score'])
        previous_rating = float(request.form['previous_year_rating'])
        awards_won = int(request.form['awards_won'])

        # Enhanced validation with specific error messages
        error_messages = []
        if department == -1:
            error_messages.append(f"Department '{department_raw}' not recognized. Please select from the dropdown options.")
        if job_role == -1:
            error_messages.append(f"Job role '{job_role_raw}' not recognized. Try: manager, analyst, developer, executive, associate, intern, consultant, or combinations like 'sales manager', 'data analyst', etc.")
        if gender == -1:
            error_messages.append(f"Gender '{gender_raw}' not recognized. Please select from the dropdown options.")
            
        if error_messages:
            error_text = "Validation errors: " + "; ".join(error_messages)
            return render_template("index.html", prediction_text=error_text)

        # Create feature array for prediction
        features = np.array([[education, age, department, job_role, experience, gender,
                              training_score, previous_rating, awards_won]])

        # Make prediction
        prediction = model.predict(features)
        
        # Format the prediction result nicely
        prediction_value = prediction[0]
        if isinstance(prediction_value, (int, float)):
            if prediction_value >= 4:
                result_text = f"High Performance Predicted (Score: {prediction_value:.2f}) - This employee shows excellent potential!"
            elif prediction_value >= 3:
                result_text = f"Good Performance Predicted (Score: {prediction_value:.2f}) - This employee shows solid performance indicators."
            else:
                result_text = f"Average Performance Predicted (Score: {prediction_value:.2f}) - Consider additional training and support."
        else:
            result_text = f"Predicted Performance Category: {prediction_value}"
        
        return render_template('index.html', prediction_text=result_text)

    except KeyError as e:
        return render_template("index.html", prediction_text=f"Missing form field: {str(e)}")
    except ValueError as e:
        return render_template("index.html", prediction_text=f"Invalid number format: {str(e)}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)