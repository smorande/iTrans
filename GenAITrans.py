import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import requests
import openai
import time
from typing import Tuple, Dict

class SmartScriptOCR:
    def __init__(self):
        self.supported_languages = {
            'Hindi': ['hi'],
            'English': ['en'],
            'Arabic': ['ar']
        }
        self.easy_reader = None
        openai.api_key = st.secrets["openai_api_key"]
        self.xai_api_key = st.secrets["xai_api_key"]
        
    def initialize_ocr(self, lang: str) -> None:
        lang_code = self.supported_languages[lang][0]
        self.easy_reader = easyocr.Reader([lang_code], gpu=False)
        
    def process_genai(self, text: str) -> str:
        try:
            grok_url = st.secrets["xai_api_url"]
            headers = {
                'Authorization': f'Bearer {self.xai_api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'messages': [
                    {'role': 'assistant', 'content': 'You are an OCR enhancement assistant.'},
                    {'role': 'user', 'content': f'Enhance this text preserving language: {text}'}
                ]
            }
            
            response = requests.post(grok_url, headers=headers, json=payload, verify=True)
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            
            # Fallback to OpenAI
            response = openai.chat.completions.create(
                model=st.secrets["openai_model"],
                messages=[
                    {"role": "system", "content": "Enhance OCR text while preserving language."},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"GenAI Error: {str(e)}")
            return text

def create_ui():
    st.set_page_config(
        page_title="SmartScript: AI-Powered Handwriting Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        
        .main {
            background-color: #ffffff;
            font-family: 'Inter', sans-serif;
            border: 2px solid #e5e7eb;
            border-radius: 16px;
            padding: 24px;
            margin: 16px;
        }
        .stButton>button {
            background-color: #60a5fa;
            color: white;
            font-weight: 500;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            transition: all 0.2s ease-in-out;
            letter-spacing: 0.3px;
            border: none;
            font-size: 15px;
            box-shadow: 0 2px 4px rgba(96, 165, 250, 0.1);
        }
        .stButton>button:hover {
            background-color: #3b82f6;
            box-shadow: 0 4px 6px rgba(96, 165, 250, 0.2);
        }
        .success-metric {
            padding: 20px;
            border-radius: 12px;
            margin: 12px 0;
            border: 1px solid rgba(96, 165, 250, 0.1);
            background-color: #f0f7ff;
            box-shadow: 0 2px 8px rgba(96, 165, 250, 0.05);
        }
        .qa-section {
            background-color: #f0f7ff;
            padding: 24px;
            border-radius: 12px;
            margin-top: 24px;
            border: 1px solid #93c5fd;
        }
        .result-text {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
            margin: 16px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .analyze-button-container {
            margin-top: -4cm !important;
            padding: 20px;
            text-align: center;
        }
        .stSelectbox {
            background-color: white;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        }
        .stRadio > label {
            color: #334155;
            font-weight: 500;
        }
        .stFileUploader {
            border: 2px dashed #93c5fd;
            border-radius: 12px;
            padding: 20px;
            background-color: #f8fafc;
        }
        .stSpinner {
            color: #60a5fa !important;
        }
        .stAlert {
            background-color: #f0f7ff;
            border-color: #93c5fd;
            color: #1e40af;
        }
        div[data-testid="stVerticalBlock"] {
            padding: 0 16px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('''<h1 style="color: #3b82f6; text-align: center; font-size: 2.5em; font-weight: 600">✨ SmartScript</h1>''', unsafe_allow_html=True)
    st.markdown('''<h4 style="text-align: center; color: #64748b; font-weight: 400">Transform handwritten documents into digital text with AI precision</h4>''', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📝 OCR Processing", "🤖 Q&A Analysis"])
    
    with tab1:
        ocr_system = SmartScriptOCR()
        col1, col2 = st.columns([2,1])
        
        with col1:
            selected_lang = st.selectbox(
                "Select Language / भाषा चुनें / اختر اللغة",
                list(ocr_system.supported_languages.keys()),
                index=0
            )
            mode = st.radio(
                "Processing Mode",
                ["Traditional ML", "GenAI Enhanced"],
                index=1
            )
            uploaded_file = st.file_uploader("Upload Handwritten Document", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            with col2:
                st.image(image, "Uploaded Document", use_container_width=True)
            
            st.markdown("<div class='analyze-button-container'>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                if st.button("🔍 Analyze Document"):
                    with st.spinner("Processing..."):
                        try:
                            start_time = time.time()
                            ocr_system.initialize_ocr(selected_lang)
                            
                            img_array = np.array(image)
                            processed_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
                            text, confidence = ocr_system.process_traditional(processed_img)
                                
                            if mode == "GenAI Enhanced":
                                text = ocr_system.process_genai(text)
                            
                            processing_time = round(time.time() - start_time, 2)
                            
                            st.markdown("<div style='margin-top: 24px'><h3 style='color: #3b82f6'>📝 Extracted Text</h3></div>", unsafe_allow_html=True)
                            st.markdown(f'<div class="result-text">{text}</div>', unsafe_allow_html=True)
                            
                            st.markdown("<div style='margin-top: 32px'><h3 style='color: #3b82f6'>📊 Analysis Results</h3></div>", unsafe_allow_html=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"""
                                    <div class='success-metric'>
                                        <div style='color: #3b82f6; font-weight: 500'>⏱️ Processing Time</div>
                                        <div style='font-size: 1.5em; color: #60a5fa; margin-top: 8px'>{processing_time}s</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            with col2:
                                st.markdown(f"""
                                    <div class='success-metric'>
                                        <div style='color: #3b82f6; font-weight: 500'>📈 Confidence Score</div>
                                        <div style='font-size: 1.5em; color: #60a5fa; margin-top: 8px'>{confidence}%</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            st.session_state['processed_text'] = text
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        if 'processed_text' in st.session_state:
            st.markdown("<h3 style='color: #3b82f6'>🤖 Ask Questions About Your Document</h3>", unsafe_allow_html=True)
            question = st.text_input("Enter your question:", placeholder="Type your question here...")
            
            if question and st.button("Get Answer"):
                try:
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Answer questions based on the provided text."},
                            {"role": "user", "content": f"Text: {st.session_state['processed_text']}\nQuestion: {question}"}
                        ]
                    )
                    answer = response.choices[0].message.content
                    st.markdown(f"""
                        <div class='qa-section'>
                            <div style='color: #3b82f6; font-weight: 500; margin-bottom: 12px'>Q: {question}</div>
                            <div style='color: #334155'>A: {answer}</div>
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
        else:
            st.info("Please process a document first in the OCR Processing tab.")

if __name__ == "__main__":
    create_ui()
    
    #Devloped by Dr. Swapnil M.