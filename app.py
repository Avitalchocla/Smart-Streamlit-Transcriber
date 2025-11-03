import streamlit as st
import requests
import tempfile
import os
import time
from typing import Optional, Tuple
from google.cloud import speech 
import io

# הגדרת אורך מקסימלי לקובץ עבור Hugging Face
HF_MAX_SIZE_MB = 25 

class FreeTranscriber:
    def __init__(self):
        # סדר עדיפות: AssemblyAI, HuggingFace, Google
        self.providers = ["assemblyai", "huggingface", "google"]
    
    # --- פונקציה 1: Hugging Face (חינם) ---
    def transcribe_huggingface(self, audio_file_path) -> Optional[str]:
        """Hugging Face Whisper - מתאים לקבצים קצרים."""
        try:
            file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
            if file_size_mb > HF_MAX_SIZE_MB:
                st.warning(f"🤖 Hugging Face דורש קבצים קטנים מ-{HF_MAX_SIZE_MB}MB. מדלג.")
                return None
            if not st.secrets.get('HF_TOKEN'):
                return None

            API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
            headers = {"Authorization": f"Bearer {st.secrets.get('HF_TOKEN')}"}
            
            with open(audio_file_path, "rb") as f:
                data = f.read()
            
            response = requests.post(API_URL, headers=headers, data=data)
            result = response.json()
            
            if 'text' in result:
                return result['text']
            return None
        except Exception:
            return None

    # --- פונקציה 2: AssemblyAI (מהיר ואמין) ---
    def transcribe_assemblyai(self, audio_file_path) -> Optional[str]:
        """AssemblyAI - תמלול אסינכרוני עם כפיית עברית."""
        try:
            if not st.secrets.get('ASSEMBLYAI_TOKEN'):
                return None
                
            headers = {
                "authorization": st.secrets.get('ASSEMBLYAI_TOKEN'),
                "content-type": "application/json"
            }
            
            status_placeholder = st.empty() 
            status_placeholder.info("🔄 AssemblyAI: מעלה קובץ לשרת...")
            
            # 1. העלאה ל-AssemblyAI
            upload_url = "https://api.assemblyai.com/v2/upload"
            with open(audio_file_path, "rb") as f:
                upload_response = requests.post(upload_url, headers=headers, data=f)
            upload_data = upload_response.json()
            
            if 'error' in upload_data:
                status_placeholder.error(f"❌ AssemblyAI כשל בהעלאה: {upload_data['error']}")
                return None

            # 2. תחילת תימלול
            transcript_url = "https://api.assemblyai.com/v2/transcript"
            
            data = {
                "audio_url": upload_data["upload_url"],
                "language_code": "he" # כפיית עברית
            }
                
            transcript_response = requests.post(
                transcript_url,
                headers=headers,
                json=data
            )
            transcript_data = transcript_response.json()
            
            if 'error' in transcript_data:
                status_placeholder.error(f"❌ AssemblyAI כשל ביצירת משימה: {transcript_data['error']}")
                return None
            
            # 3. המתנה להשלמה (Polling)
            polling_url = f"https://api.assemblyai.com/v2/transcript/{transcript_data['id']}"
            
            while True:
                polling_response = requests.get(polling_url, headers=headers)
                polling_data = polling_response.json()
                
                current_status = polling_data.get("status", "unknown")
                status_placeholder.info(f"🎧 AssemblyAI: סטאטוס - **{current_status.upper()}** (ID: {transcript_data['id']})")
                
                if current_status == "completed":
                    status_placeholder.success("✅ AssemblyAI: התימלול הושלם. מציג תוצאות...")
                    status_placeholder.empty() 
                    return polling_data["text"] 
                    
                elif current_status == "error":
                    status_placeholder.error(f"❌ AssemblyAI: שגיאה בתימלול: {polling_data.get('error')}")
                    return None
                
                time.sleep(5) 
                
        except Exception:
            return None

    # --- פונקציה 3: Google Speech-to-Text (סינכרוני) ---
    def transcribe_google(self, audio_file_path) -> Optional[str]:
        """Google Speech-to-Text - חינמי 60 דקות/חודש."""
        try:
            client = speech.SpeechClient()
            status_placeholder = st.empty()
            
            with io.open(audio_file_path, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                language_code="he-IL", # כפיית עברית
            )
            
            # בדיקה גסה של גודל קובץ למניעת כשל ב-API סינכרוני
            if len(content) > 10 * 1024 * 1024: 
                 status_placeholder.warning("☁️ קובץ גדול מדי לתמלול סינכרוני בגוגל. מדלג.")
                 return None
            
            status_placeholder.info("☁️ Google: מתמלל קובץ...")
            response = client.recognize(config=config, audio=audio)
            status_placeholder.empty()
            
            text = " ".join([result.alternatives[0].transcript for result in response.results])
            return text.strip() if text else None
            
        except Exception as e:
            st.warning(f"☁️ Google Speech-to-Text נכשל. ודא אימות: {str(e)}")
            return None

    # --- פונקציה ראשית: Smart Transcribe ---
    def smart_transcribe(self, audio_file_path) -> Tuple[str, str]:
        """מנסה את כל ה-APIs לפי סדר עדיפות."""
        st.info("🔍 מחפש את השירות המהיר והמדויק ביותר...")
        
        for provider in self.providers:
            st.write(f"🔄 מנסה {provider}...")
            
            result = None
            if provider == "assemblyai":
                result = self.transcribe_assemblyai(audio_file_path)
            elif provider == "huggingface":
                result = self.transcribe_huggingface(audio_file_path)
            elif provider == "google":
                result = self.transcribe_google(audio_file_path)
            
            if result:
                st.success(f"✅ הצלחה עם {provider}!")
                return result, provider
        
        return "❌ כל השירותים נכשלו. ודא ש-API Keys נכונים.", "none"

# 🎨 ממשק משתמש (Streamlit Frontend)
def main():
    st.set_page_config(page_title="🎤 פלטפורמת תימלול שמע חכמה", page_icon="🎵")
    
    # --- CSS מותאם אישית לרקע תכלת ---
    st.markdown("""
        <style>
        .stApp {
            background-color: #F0F8FF; /* Alice Blue - תכלת בהיר */
        }
        .main-header {
            color: #1E90FF; /* כחול רויאל לכותרת */
            text-align: right;
            border-bottom: 2px solid #ADD8E6; /* קו תחתון תכלת */
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # --- כותרת מכובדת ---
    st.markdown('<h1 class="main-header">🎤 מתמלל חכם</h1>', unsafe_allow_html=True)
    st.markdown("### נבנה ע\"י א. קצבורג")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("העלה קובץ שמע", type=['mp3', 'wav', 'm4a', 'ogg', 'mp4'])
    
    if uploaded_file and st.button("🎯 התחל תימלול חכם"):
        
        # שמירת הקובץ הזמני
        file_extension = uploaded_file.name.split('.')[-1]
        
        temp_dir = 'temp_uploads'
        os.makedirs(temp_dir, exist_ok=True)
        audio_path = os.path.join(temp_dir, f"temp_audio.{file_extension}")

        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        
        try:
            transcriber = FreeTranscriber()
            result, provider = transcriber.smart_transcribe(audio_path)
        finally:
            # ניקוי הקובץ הזמני תמיד
            os.unlink(audio_path)
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        
        # הצגת התוצאה
        st.subheader("📄 תוצאות התימלול:")
        
        if "❌" in result:
             st.error(result)
        else:
            st.markdown(result) 
            st.info(f"**שותף ששימש לתמלול:** {provider}")
        
            st.download_button(
                label="📥 הורד כקובץ טקסט",
                data=result,
                file_name="transcript.txt",
                mime="text/plain"
            )

# 🚀 הפעלה
if __name__ == "__main__":
    main()
