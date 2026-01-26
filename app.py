
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

# Streamlit app layout
st.title('XAI Word level Aphasia Prediction')

# Provide instructions for the user
st.write('Please provide the input values below, and the model will predict the output.')

# Create a dictionary to hold the input values
input_values = {}

# Mapping feature names to user-friendly labels
feature_labels = {
    'MPO': 'Months Since Stroke',
    'Yrs Edu': 'Years of Education',
    'Age': 'Age',
    'NWF_WAB_Avg': 'Western Aphasia Battery Naming Subscore (0-10)',
    'Avg_WAB_AQ ': 'Western Aphasia Battery Aphasia Quotient Score (0-100)',
    'Lesion_Volume': 'Lesion Volume',
    'Freq_Cond': 'Frequency Condition (High/Low)',
    'Syllables_avg (SyllaPy)': 'Syllables in Word',
    'Phonemes_avg (CMUDict)': 'Phonemes in Word'
}

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
        word_input = st.text_input("Enter a word to determine Frequency Condition")
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
if st.button('Make Prediction'):
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
    st.write(f'Prediction: {result}')

    # Display the model's confidence (probability)
    st.write(f'Model Confidence (Probability): {prediction_proba[0][prediction[0]]:.4f}')


    # Now, we create the SHAP explainer and plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_features)

    # Plot SHAP summary plot
    st.subheader('Feature Importance Summary Plot')
    shap.summary_plot(shap_values[:,:,1], input_features, feature_names=feature_names, plot_type="bar", show=False)
    st.pyplot(plt.gcf())
