import streamlit as st
import json
from speech_processing.phonetic_analyzer import get_phonetic_analysis
# from diagnosis.rules_engine import generate_report_with_rules
from utils.util_file import load_audio

st.set_page_config(page_title="Sanskrit Pronunciation Coach", layout="wide")
st.title("Sanskrit Pronunciation Coach ")
st.markdown("Upload an audio recording of a Sanskrit word or phrase to get a detailed analysis of your pronunciation.")


ground_truth_text = st.text_input(
    "1. Enter the Sanskrit text here (in SLP1 or Devanagari):",
    "mAtApitfByAm jagato namo vAmArDajAnaye"
)
uploaded_audio = st.file_uploader(
    "2. Upload your audio recording (WAV or MP3 format):",
    type=["wav", "mp3"]
)
if st.button("Analyze Pronunciation"):
    
    if uploaded_audio is not None and ground_truth_text:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_audio.getbuffer())
        with st.spinner("Analyzing your speech... This may take a moment."):
            loaded_audio = load_audio("temp_audio.wav")
            analysis_result = get_phonetic_analysis(loaded_audio, ground_truth_text)
            # diagnostic_report_json = generate_report_with_rules(analysis_result)
        
        st.success("Analysis Complete!")
        st.subheader("Diagnostic Report")
        st.write(analysis_result)
        # report_data = json.loads(diagnostic_report_json)
        # st.write(report_data['response']['diagnostic_report'])
        # if report_data['response']['trigger_tts'] == "true":
        #     st.info("It's recommended to listen to the correct pronunciation.")

    else:
        st.error("Please enter the text and upload an audio file before analyzing.")
        
    
    