import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib
import shap
import syllapy
import nltk
from nltk.corpus import cmudict
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
nltk.download('cmudict')

# Load the trained Random Forest model
model = joblib.load('simple_rf_best_model.joblib')

# Load the backend list of words for Freq_Cond calculation
word_list = pd.read_csv('word_list.csv')['word'].tolist()

# Load the CMU Pronouncing Dictionary
cmu_dict = cmudict.dict()

# Get the feature names from the model
feature_names = model.feature_names_in_

# Simple sidebar
st.sidebar.title("App versions menu")

page = st.sidebar.radio(
    "Navigate",
    ["Single Subject", "Multiple Subjects", "Feature Names Explained"]
)
# Single Subject Page
if page == "Single Subject Demo with Explainability":
    
    # Streamlit app layout
    st.title('AphasiaLENS (Lexical Estimator of Naming in Speech)')
    st.write('Word-by-word personalized predictions of naming ability in chronic post stroke aphasia, using clinically available inputs and explainable machine learning')
    st.write('Disclaimer: This web application is intended for research, education, and demo purposes. Please do not use it for medical advice, diagnosis, or treatment without consulting professional medical advice.')
                
    # instructions for the user
    st.divider()
    st.subheader('Please input the information of the person with aphasia')

    # Create a dictionary to hold the input values
    input_values = {}

    # Mapping feature names to user-friendly labels
    feature_labels = {
        'MPO': 'Months Since Stroke',
        'Yrs Edu': 'Years of Education (counted from first grade)',
        'Age': 'Current Age (in years, range 35-95)',
        'NWF_WAB_Avg': 'Western Aphasia Battery Naming Subscore (0-10)',
        'Avg_WAB_AQ ': 'Western Aphasia Battery Aphasia Quotient Score (0-100)',
        'Lesion_Volume': 'Lesion Volume (range from none to entire left hemisphere)',
        'Freq_Cond': 'Frequency Condition (High/Low)',
        'Syllables_avg (SyllaPy)': 'Syllables in Word',
        'Phonemes_avg (CMUDict)': 'Phonemes in Word'
    }

    feature_names_shap = ['Months Since Stroke','Years of Education','Age','WAB Naming Subscore','WAB Aphasia Quotient','Lesion_Volume',
                        'Word Frequency (High/Low)','# of Syllables', '# of Phonemes']

    # Loop through each feature and create the appropriate input field
    for feature in feature_names:
        # If the feature is 'MPO', 'Yrs Edu', 'Age', 'Lesion_Volume', we use a number input
        if feature in ['MPO', 'Yrs Edu', 'Age']:
            input_values[feature] = st.number_input(feature_labels[feature], value=0)


        # If the feature is 'NWF_WAB_Avg' or 'Avg_WAB_AQ' or Lesion Volume , we use sliders
        elif feature == 'NWF_WAB_Avg':
            input_values[feature] = st.slider(feature_labels[feature], 0, 10, 5)
        elif feature == 'Avg_WAB_AQ ':
            input_values[feature] = st.slider(feature_labels[feature], 0, 100, 50)

        elif feature == 'Lesion_Volume':
            input_values[feature] = st.slider(feature_labels[feature], 0, 500000, 0)

        # If the feature is 'Freq_Cond', we ask the user to input a word
        elif feature == 'Freq_Cond':
            word_input = st.text_input("Enter the word to be spoken")
            if word_input:
                # Compute frequency condition based on whether word is in the backend list
                if word_input.lower() in [w.lower() for w in word_list]:
                    input_values[feature] = 'High'
                else:
                    input_values[feature] = 'Low'

        # If the feature is 'Syllables_avg (SyllaPy)' or 'Phonemes_avg (CMUDict)', we compute from the input word
        elif feature == 'Syllables_avg (SyllaPy)' or feature == 'Phonemes_avg (CMUDict)':
            if word_input:
                # Compute syllables using syllapy
                syllables_count = syllapy.count(word_input)
                # Compute phonemes using CMUdict
                phonemes = cmu_dict.get(word_input.lower())
                phonemes_count = len(phonemes[0]) if phonemes else 0
                if feature == 'Syllables_avg (SyllaPy)':
                    input_values[feature] = syllables_count
                else:
                    input_values[feature] = phonemes_count

    # Button to trigger prediction
    if st.button('Predict Speech Accuracy'):
        if not word_input or word_input.strip() == "":
            st.warning("Please enter a word before making a prediction.")
            st.stop()
        # Prepare the input data for prediction
        encoded_features = []

        for feature in feature_names:
            value = input_values[feature]

            # Convert categorical features to numerical values (e.g., 'High' and 'Low' for Freq_Cond)
            if isinstance(value, str):  # Categorical features will be string
                if feature == 'Freq_Cond':
                    value = 1 if value == 'High' else 0
                # Other categorical conversions (if needed) can go here

            # Append the feature value (either categorical or numerical)
            encoded_features.append(value)

        # Convert list to numpy array and reshape for prediction
        input_features = np.array(encoded_features).reshape(1, -1)

        # Make the prediction using the Random Forest model
        prediction = model.predict(input_features)

        # Get the prediction probabilities
        prediction_proba = model.predict_proba(input_features)

        # Display the result as "Correct" or "Wrong"
        result = "Correct" if prediction[0] == 1 else "Wrong"

        # Display the result
        st.divider()
        st.subheader('Speech Accuracy Prediction for this Word')
        st.write(f'Prediction: {result}')

        # Display the model's confidence (probability)
        st.write(f'Model Confidence (Probability): {prediction_proba[0][prediction[0]]:.4f}')
        st.divider()

        # Now, we create the SHAP explainer and plot
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_features)

        # Plot SHAP summary plot
        st.subheader('Feature Importances for this Prediction')
        shap.summary_plot(shap_values[:,:,1], input_features, feature_names=feature_names_shap, plot_type="bar", show=False)
        st.pyplot(plt.gcf())

# Batched Version of App (without shap plots)
elif page == "Batch Processing Multiple PWA and Multiple Words":
    
    # Streamlit app layout
    st.title('Batched Version of AphasiaLENS')
    st.write('Generate model predictions of word-level accuracy and prediction probability, for large lists of patients and words at once.')
    st.write('Disclaimer: This web application is intended for research, education, and demo purposes. Please do not use it for medical advice, diagnosis, or treatment without consulting professional medical advice.')
                
    # instructions for the user
    st.divider()
    st.subheader('Please upload a CSV file of clinical information for each subject')

    # Download template
    # Random example values
    template = pd.DataFrame({
        "Subject": ["SubjID-1"],
        "MPO": [12],
        "Yrs Edu": [16],
        "Age": [68],
        "NWF_WAB_Avg": [8],
        "Avg_WAB_AQ ": [85],
        "Lesion_Volume": [12000]
    })

    # Clinical csv for download
    st.download_button(
        "Download Template of Clinical CSV",
        template.to_csv(index=False),
        "clinical_data.csv",
        "text/csv"
    )

    uploaded_clinical_file = st.file_uploader("Upload Clinical CSV", type=["csv"])

    # instructions for the user
    st.divider()
    st.subheader('Please upload a CSV file of all trials')

    # Download template
    # Random example values
    template = pd.DataFrame({
        "Subject": ["SubjID-1"],
        "Word": ["apple"],
    })

    # Trial csv for download
    st.download_button(
        "Download Template of Trials CSV",
        template.to_csv(index=False),
        "trial_data.csv",
        "text/csv"
    )

    uploaded_trial_file = st.file_uploader("Upload Trial CSV", type=["csv"])

    st.divider()

    # Only enable button when both files have been uploaded
    can_predict = (uploaded_clinical_file is not None and uploaded_trial_file is not None)

    # Check for errors in the uploaded files
    error = False

    # Button to trigger prediction
    if st.button("Predict Speech Accuracy", disabled=not can_predict):

        clinical_dataset = pd.read_csv(uploaded_clinical_file)
        trial_dataset = pd.read_csv(uploaded_trial_file)
        required_clinical_columns = ['Subject', 'MPO', 'Yrs Edu', 'Age', 'NWF_WAB_Avg', 'Avg_WAB_AQ ', 'Lesion_Volume']
        required_trial_columns = ['Subject', 'Word']

        # Missing Values
        if clinical_dataset.isna().any().any():
            st.warning("Missing values detected in the clinical dataset.")
            error = True
        if trial_dataset.isna().any().any():
            st.warning("Missing values detected in the trial dataset.")
            error = True

        # Negative Values
        if (clinical_dataset.select_dtypes(include='number') < 0).any().any():
            st.warning("Negative values detected in the clinical dataset.")
            error = True
        if (trial_dataset.select_dtypes(include='number') < 0).any().any():
            st.warning("Negative values detected in the trial dataset.")
            error = True
        
        # Check for missing clinical columns
        missing_clinical_columns = [col for col in required_clinical_columns if col not in clinical_dataset.columns]
        if missing_clinical_columns:
            st.warning(f"Missing required columns in the clinical dataset: {', '.join(missing_clinical_columns)}.")
            error = True

        # Check for missing trial columns
        missing_trial_columns = [col for col in required_trial_columns if col not in trial_dataset.columns]
        if missing_trial_columns:
            st.warning(f"Missing required columns in the trial dataset: {', '.join(missing_trial_columns)}.")
            error = True

        # Check for extra clinical columns
        extra_clinical_columns = [col for col in clinical_dataset.columns if col not in required_clinical_columns]
        if extra_clinical_columns:
            st.warning(f"Unexpected extra columns found in the clinical dataset: {', '.join(extra_clinical_columns)}.")
            error = True

        # Check for extra trial columns
        extra_trial_columns = [col for col in trial_dataset.columns if col not in required_trial_columns]
        if extra_trial_columns:
            st.warning(f"Unexpected extra columns found in the trial dataset: {', '.join(extra_trial_columns)}.")
            error = True

        # Check that all subjects in trials have a corresponding name in clinical data
        trial_subjects = set(trial_dataset['Subject'])
        clinical_subjects = set(clinical_dataset['Subject'])

        if not trial_subjects.issubset(clinical_subjects):
            missing_subjects = trial_subjects - clinical_subjects
            st.warning(f"The following subjects in the trial dataset are missing from the clinical dataset: {', '.join(map(str, missing_subjects))}")
            error = True

        # Check that all columns except 'Subject' in clinical data are numerical
        cols_to_check = [col for col in clinical_dataset.columns if col != 'Subject']
        non_numeric_clinical_cols = clinical_dataset[cols_to_check].select_dtypes(exclude='number').columns.tolist()

        if non_numeric_clinical_cols:
            st.warning(f"The following clinical columns contain non-numerical data: {', '.join(non_numeric_clinical_cols)}.")
            error = True

        # Dataset of clinical and trial data merged on Subject ID
        X_clin_ling = clinical_dataset.merge(trial_dataset, on='Subject', how='inner')
        # remove any leading/trailing whitespace and convert to lowercase
        X_clin_ling['Word'] = X_clin_ling['Word'].str.lower().str.strip()

        # Check for weird characters in the Word column
        if 'Word' in X_clin_ling.columns:
            invalid_words_mask = X_clin_ling['Word'].astype(str).str.contains(r'[^a-zA-Z]', regex=True)
            if invalid_words_mask.any():
                invalid_samples = X_clin_ling.loc[invalid_words_mask, 'Word'].unique()
                st.warning("Invalid characters (spaces, punctuation, or numbers) detected in the 'Word' column.")
                error = True

        if error:
            st.stop()

        # Columns for derived features
        freq_cond_list = []
        syllables_list = []
        phonemes_list = []
        
        # Lowercase word list 
        word_list_lower = set(w.lower() for w in word_list)

        # for each word, add derived features
        for word in X_clin_ling['Word']:
            # Compute frequency condition
            freq_cond_list.append(1 if word in word_list_lower else 0)

            # Compute syllables
            syllables_list.append(syllapy.count(word))

            # Compute Phonemes_avg (CMUDict) safely
            phonemes = cmu_dict.get(word.lower())
            phonemes_list.append(len(phonemes[0]) if phonemes and len(phonemes) > 0 else 0)

        # add on the derived feature columns
        X_clin_ling['Freq_Cond'] = freq_cond_list
        X_clin_ling['Syllables_avg (SyllaPy)'] = syllables_list
        X_clin_ling['Phonemes_avg (CMUDict)'] = phonemes_list

        # Extract all the features in model
        X_pred_all = X_clin_ling[feature_names]

        #run all predictions & probabilites
        predictions = model.predict(X_pred_all)
        probabilities = model.predict_proba(X_pred_all)
            
        pred_y = X_clin_ling

        # Assign prediction results and model confidence back to the dataframe efficiently
        pred_y['Prediction'] = ["Correct" if p == 1 else "Wrong" for p in predictions]
        pred_y['Model Confidence (Probability)'] = [
            f"{prob[pred]:.4f}" for prob, pred in zip(probabilities, predictions)
        ]

        st.divider()

        # Preview of the predictions
        st.subheader('Preview of Predictions (Top 5 Rows)')
        st.dataframe(pred_y.head(5))

        st.divider()

        st.subheader('Download Predictions CSV')

        # Convert DataFrame to CSV
        csv_pred_y = pred_y.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv_pred_y,
            file_name="Predicted_Results.csv",
            mime="text/csv"
        )


# Full feature names explained
elif page == "Feature Names Explained":

    st.title("Feature Names Explained")

    features = {
        "Subject": ("Subject ID", "String"),
        "Word": ("Predicted Word", "String"),
        "MPO": ("Months Since Stroke", "Numeric"),
        "Yrs Edu": ("Years of Education (counted from first grade)", "Numeric"),
        "Age": ("Current Age (in years, range 35-95)", "Numeric"),
        "NWF_WAB_Avg": ("Western Aphasia Battery Naming Subscore (0-10)", "Numeric"),
        "Avg_WAB_AQ ": ("Western Aphasia Battery Aphasia Quotient Score (0-100)", "Numeric"),
        "Lesion_Volume": ("Lesion Volume (range from none to entire left hemisphere)", "Numeric"),
        "Freq_Cond": ("Frequency Condition (High/Low)", "Derived"),
        "Syllables_avg (CMU_dict)": ("Syllables in Word", "Derived"),
        "Phonemes_avg (CMU_dict)": ("Phonemes in Word", "Derived"),
    }

    for k, v in features.items():
        st.markdown(f"**{k}**")
        st.write(f"- Full Name: {v[0]}")
        st.write(f"- Type: {v[1]}")
        st.divider()
