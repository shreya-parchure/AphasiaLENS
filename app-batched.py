import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib
import shap
import pronouncing
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
st.sidebar.title("AphasiaLENS Menu")

page = st.sidebar.radio(
    "Navigate",
    ["Single Subject", "Multiple Subjects", "Feature Names Explained"]
)
# Single Subject Page
if page == "Single Subject":

    # Streamlit app layout
    st.title('AphasiaLENS (Lexical Estimator of Naming in Speech)')
    st.write('Word-by-word personalized predictions of naming ability in chronic post stroke aphasia, using clinically available inputs and explainable machine learning')
    st.write('Disclaimer: This web application is intended for research, education, and demo purposes. Please do not use it for medical advice, diagnosis, or treatment without consulting professional medical advice.')
                
    # Instructions for the user
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


        # If the feature is 'NWF_WAB_Avg' or 'Avg_WAB_AQ' or Lesion VOlume , we use sliders
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
                phones = pronouncing.phones_for_word(word_input.lower())
                syllables_count = pronouncing.syllable_count(phones)
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

        # Convert list to DataFrame with feature names to match model training
        input_features = pd.DataFrame([encoded_features], columns=feature_names)

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
elif page == "Multiple Subjects":

    st.title("Batch Mode")

    st.write(
        "Upload a CSV with Subject × Word rows. "
        "The model will compute lexical + clinical predictions."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    # Download template
    st.subheader("Download Template CSV")

    # Random example values
    template = pd.DataFrame({
        "Subject": ["SubjID-1"],
        "Word": ["apple"],
        "MPO": [12],
        "Yrs Edu": [16],
        "Age": [68],
        "NWF_WAB_Avg": [8],
        "Avg_WAB_AQ ": [85],
        "Lesion_Volume": [12000]
    })

    # csv for download
    st.download_button(
        "Download Template CSV",
        template.to_csv(index=False),
        "aphasia_template.csv",
        "text/csv"
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        if "Word" not in df.columns:
            st.error("Missing required column: Word")
            st.stop()

        words = df["Word"].astype(str)


        # Calculate high/low Freq_Cond
        df["Freq_Cond"] = np.where(
            words.str.lower().isin(set(w.lower() for w in word_list)),
            "High",
            "Low"
        )

        # Calculate the Syllables_avg (CMU_dict)
        df["Syllables_avg (CMU_dict)"] = words.apply(lambda w: pronouncing.syllable_count(pronouncing.phones_for_word(w.lower())))

        def phoneme_count(w):
            ph = cmu_dict.get(w.lower())
            return len(ph[0]) if ph else 0
        
        # Calculate the Phonemes_avg (CMUDict)
        df["Phonemes_avg (CMUDict)"] = words.apply(phoneme_count)


        model_df = df.copy()

        # Convert from high/low to 1/0 for the model
        model_df["Freq_Cond"] = model_df["Freq_Cond"].map({"High": 1, "Low": 0})

        # Create the feature matrix for prediction
        X = model_df[feature_names]

        # Prediction Column
        preds = model.predict(X)

        # Confidence Column
        probs = model.predict_proba(X)

        # Write out Correct/Wrong for prediction column
        df["Prediction"] = np.where(preds == 1, "Correct", "Wrong")

        # Write out the confidence for each prediction
        df["Confidence"] = [probs[i, p] for i, p in enumerate(preds)]

        # display success if successfully processed
        st.success(f"Processed {len(df)} rows")

        st.dataframe(df, use_container_width=True)

        # Download the rsults as a CSV file
        st.download_button(
            "Download Output CSV",
            df.to_csv(index=False),
            "aphasia_batch_results.csv",
            "text/csv"
        )

    else:

        # while nothing is uploaded
        st.info("Upload CSV to run batch inference")


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
