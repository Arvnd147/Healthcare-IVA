# healthcare_iva_app.py

import streamlit as st
import numpy as np
import skfuzzy as fuzz
from datasets import load_dataset
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd  # For data manipulation and visualization

# ===========================
# 1. Application Setup
# ===========================

# Set the title and description of the app
st.title("🩺 Healthcare Intelligent Virtual Assistant (IVA)")
st.write("""
Provide your symptoms and severity to receive preliminary diagnoses and recommendations.
This IVA leverages Natural Language Processing (NLP), Fuzzy Logic, and Vector Databases to assist you.
""")

# ===========================
# 2. Caching Functions
# ===========================

@st.cache_data
def load_medical_data():
    """
    Load the medical symptom dataset.
    
    Returns:
        Dataset: The loaded dataset containing medical information.
    """
    dataset = load_dataset("mohammad2928git/complete_medical_symptom_dataset")
    symptoms_data = dataset['train']
    return symptoms_data

@st.cache_resource
def load_nlp_model():
    """
    Load a pre-trained NLP model for symptom interpretation.
    
    Returns:
        Pipeline: The loaded NLP question-answering pipeline.
    """
    nlp = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    return nlp

@st.cache_resource
def load_embedding_model():
    """
    Load SentenceTransformer for vector embeddings.
    
    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

import os

# Define the FAISS index file path
FAISS_INDEX_FILE = "faiss_index.bin"

@st.cache_resource
def load_faiss_index_or_create(symptom_descriptions,_embedding_model):
    """
    Load FAISS index from file if it exists; otherwise, create and save it.
    
    Args:
        symptom_descriptions (list of str): List of symptom descriptions.
        embedding_model (SentenceTransformer): Pre-loaded embedding model.
    
    Returns:
        tuple: FAISS index and symptom embeddings.
    """
    # Check if the index file exists
    if os.path.exists(FAISS_INDEX_FILE):
        # Load the saved FAISS index
        index = faiss.read_index(FAISS_INDEX_FILE)
        # No need to generate embeddings again if the index is loaded
        return index, None
    else:
        # Create a new FAISS index if file not found
        symptom_embeddings = embedding_model.encode(symptom_descriptions, convert_to_numpy=True)
        dimension = symptom_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(symptom_embeddings)
        
        # Save the index to the file
        faiss.write_index(index, FAISS_INDEX_FILE)
        
        return index, symptom_embeddings

# ===========================
# 3. Load Data and Models
# ===========================

# Load data and models with error handling
try:
    symptoms_data = load_medical_data()
except Exception as e:
    st.error(f"Error loading medical data: {e}")
    st.stop()

try:
    nlp_model = load_nlp_model()
except Exception as e:
    st.error(f"Error loading NLP model: {e}")
    st.stop()

try:
    embedding_model = load_embedding_model()
except Exception as e:
    st.error(f"Error loading embedding model: {e}")
    st.stop()

# ===========================
# 4. Dataset Information
# ===========================

st.subheader("📊 Dataset Information")
st.write("**Available Columns:**")
st.write(symptoms_data.column_names)

# Display a sample row to understand the data structure
if st.checkbox("🔍 Show a sample data row"):
    sample_index = st.number_input("Select row index to view", min_value=0, max_value=len(symptoms_data)-1, value=0, step=1)
    try:
        sample_row = symptoms_data[sample_index]
        st.write(sample_row)
    except Exception as e:
        st.error(f"Error fetching sample row: {e}")

# ===========================
# 5. Prepare Symptom Descriptions
# ===========================

# Choose the correct column name based on your dataset structure
symptom_column = 'symptoms'  # As per your clarification

# Verify the chosen column exists
if symptom_column not in symptoms_data.column_names:
    st.error(f"The column '{symptom_column}' does not exist in the dataset. Please choose a valid column.")
    st.stop()

# Extract symptom descriptions
# Assuming 'symptoms' is a list of symptoms per entry, we'll join them into a single string
try:
    symptom_descriptions = [
        "; ".join(symptom_list) if isinstance(symptom_list, list) else str(symptom_list)
        for symptom_list in symptoms_data[symptom_column]
    ]
except Exception as e:
    st.error(f"Error processing '{symptom_column}' column: {e}")
    st.stop()

# Load or create the FAISS index
try:
    faiss_index, symptom_embeddings = load_faiss_index_or_create(symptom_descriptions, embedding_model)
except Exception as e:
    st.error(f"Error loading or creating FAISS index: {e}")
    st.stop()

# ===========================
# 6. Define Core Functions
# ===========================

def parse_symptom(symptom_description):
    """
    Use NLP model to extract relevant symptom from the description.
    
    Args:
        symptom_description (str): User-provided symptom description.
    
    Returns:
        str: Parsed main symptom.
    """
    try:
        response = nlp_model(question="What is the main symptom?", context=symptom_description)
        return response['answer']
    except Exception as e:
        st.error(f"Error parsing symptom: {e}")
        return "Unable to parse symptom."

def fuzzy_severity_analysis(severity_score):
    """
    Analyze the severity score using fuzzy logic.
    
    Args:
        severity_score (int): Severity score between 0 and 10.
    
    Returns:
        dict: Membership values for Mild, Moderate, and Severe.
    """
    try:
        severity = np.array([severity_score])
        x_severity = np.arange(0, 11, 1)  # Severity score range from 0 to 10

        # Define fuzzy membership functions
        mild = fuzz.trimf(x_severity, [0, 0, 5])
        moderate = fuzz.trimf(x_severity, [0, 5, 10])
        severe = fuzz.trimf(x_severity, [5, 10, 10])

        # Calculate membership values
        mild_level = fuzz.interp_membership(x_severity, mild, severity)
        moderate_level = fuzz.interp_membership(x_severity, moderate, severity)
        severe_level = fuzz.interp_membership(x_severity, severe, severity)

        return {
            'Mild': round(mild_level[0], 2),
            'Moderate': round(moderate_level[0], 2),
            'Severe': round(severe_level[0], 2)
        }
    except Exception as e:
        st.error(f"Error in fuzzy severity analysis: {e}")
        return {'Mild': 0, 'Moderate': 0, 'Severe': 0}

def search_symptoms(query, embedding_model, index, symptom_descriptions, top_k=5):
    """
    Search for similar symptoms using vector embeddings and FAISS.
    
    Args:
        query (str): Parsed symptom to search for.
        embedding_model (SentenceTransformer): Pre-loaded embedding model.
        index (faiss.IndexFlatL2): FAISS index for similarity search.
        symptom_descriptions (list of str): List of symptom descriptions.
        top_k (int): Number of top similar symptoms to retrieve.
    
    Returns:
        list of str: Top similar symptoms.
    """
    try:
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, top_k)
        results = [symptom_descriptions[i] for i in indices[0]]
        return results
    except Exception as e:
        st.error(f"Error searching symptoms: {e}")
        return []

def get_preliminary_diagnosis(parsed_symptom):
    """
    Provide a preliminary diagnosis based on the parsed symptom.
    
    Args:
        parsed_symptom (str): Parsed main symptom.
    
    Returns:
        list of str: Possible diseases associated with the symptom.
    """
    try:
        # Find all entries where the parsed symptom is present
        matching_indices = [
            i for i, symptoms in enumerate(symptoms_data[symptom_column])
            if parsed_symptom.lower() in [s.lower() for s in symptoms]
        ]
        diseases = list(set([symptoms_data['lebel_text'][i] for i in matching_indices]))
        return diseases[:5]  # Return top 5 unique diseases
    except Exception as e:
        st.error(f"Error fetching preliminary diagnosis: {e}")
        return []

def healthcare_iva(symptom_description, severity_score):
    """
    Integrate all functionalities to process user input and provide outputs.
    
    Args:
        symptom_description (str): User-provided symptom description.
        severity_score (int): Severity score between 0 and 10.
    
    Returns:
        dict: Parsed symptom, severity analysis, similar symptoms, and preliminary diagnosis.
    """
    # Step 1: NLP parsing to extract relevant symptom
    parsed_symptom = parse_symptom(symptom_description)

    # Step 2: Fuzzy logic analysis for severity
    severity_analysis = fuzzy_severity_analysis(severity_score)

    # Step 3: Vector-based search for similar symptoms
    similar_symptoms = search_symptoms(parsed_symptom, embedding_model, faiss_index, symptom_descriptions)

    # Step 4: Preliminary diagnosis
    preliminary_diagnosis = get_preliminary_diagnosis(parsed_symptom)

    return {
        'Parsed Symptom': parsed_symptom,
        'Severity Analysis': severity_analysis,
        'Similar Symptoms': similar_symptoms,
        'Preliminary Diagnosis': preliminary_diagnosis
    }

# ===========================
# 7. User Interaction Tracking
# ===========================

# Initialize session state for history if not already present
if 'history' not in st.session_state:
    st.session_state.history = []

# ===========================
# 8. Streamlit Interface
# ===========================

with st.form("symptom_form"):
    st.header("📝 Enter Your Symptoms")
    symptom_description = st.text_area("Describe your symptoms:", height=150)
    severity_score = st.slider("Severity (0-10):", 0, 10, 5)
    submit_button = st.form_submit_button(label="🔍 Get Recommendations")

if submit_button:
    if symptom_description.strip() == "":
        st.warning("⚠️ Please enter your symptoms.")
    else:
        with st.spinner("🔄 Processing..."):
            result = healthcare_iva(symptom_description, severity_score)
        
        st.success("✅ Here are your results:")
        
        # Display Parsed Symptom
        st.subheader("🔍 Interpreted Symptom")
        st.write(f"**{result['Parsed Symptom'].capitalize()}**")
        
        # Display Severity Analysis
        st.subheader("📈 Severity Analysis")
        severity = result['Severity Analysis']
        severity_df = pd.DataFrame({
            'Severity Level': ['Mild', 'Moderate', 'Severe'],
            'Membership Value': [severity['Mild'], severity['Moderate'], severity['Severe']]
        })
        st.bar_chart(severity_df.set_index('Severity Level'))
        
        # Display Similar Symptoms
        st.subheader("🔗 Top Similar Symptoms")
        if result['Similar Symptoms']:
            for idx, symptom in enumerate(result['Similar Symptoms'], 1):
                st.write(f"{idx}. {symptom}")
        else:
            st.write("No similar symptoms found.")
        
        # Display Preliminary Diagnosis
        st.subheader("🩺 Preliminary Diagnosis")
        if result['Preliminary Diagnosis']:
            for idx, disease in enumerate(result['Preliminary Diagnosis'], 1):
                st.write(f"{idx}. {disease}")
        else:
            st.write("No preliminary diagnosis available.")
        
        # Update Interaction History
        st.session_state.history.append({
            'Symptom Description': symptom_description,
            'Severity Score': severity_score,
            'Parsed Symptom': result['Parsed Symptom'],
            'Severity Analysis': severity,
            'Similar Symptoms': result['Similar Symptoms'],
            'Preliminary Diagnosis': result['Preliminary Diagnosis']
        })

# Display Interaction History
if st.checkbox("📚 Show Interaction History"):
    st.subheader("🕒 Your Interaction History")
    if st.session_state.history:
        for idx, entry in enumerate(st.session_state.history, 1):
            st.markdown(f"### Entry {idx}")
            st.write(f"**Symptom Description:** {entry['Symptom Description']}")
            st.write(f"**Severity Score:** {entry['Severity Score']}")
            st.write(f"**Parsed Symptom:** {entry['Parsed Symptom']}")
            st.write(f"**Severity Analysis:** {entry['Severity Analysis']}")
            st.write(f"**Similar Symptoms:** {entry['Similar Symptoms']}")
            st.write(f"**Preliminary Diagnosis:** {entry['Preliminary Diagnosis']}")
            st.markdown("---")
    else:
        st.write("No interactions yet. Start by entering your symptoms above.")

# ===========================
# 9. Additional Frontend Enhancements
# ===========================

# (Optional) Add more visualizations or interactive elements here

# ===========================
# 10. Security and Privacy Considerations
# ===========================

# Note: Since this application handles sensitive health-related data, ensure to implement the following:
# - Do not store or log user data unless necessary and with proper consent.
# - If deploying, use secure protocols (HTTPS).
# - Comply with relevant data protection regulations (e.g., HIPAA, GDPR).

# ===========================
# 11. Deployment Considerations
# ===========================

# Once satisfied with the local version, deploying the Streamlit app using platforms like:
# - Streamlit Cloud
# - Heroku
# - AWS (Amazon Web Services)
# Ensure that the deployed environment has sufficient resources to handle model loading and FAISS indexing.