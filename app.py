import streamlit as st
import requests
import tempfile
import os
import time
from typing import Optional, Tuple
from google.cloud import speech 
import io

# הגדרת אורך מקסימלי לקובץ עבור Hugging Face (בקשות סינכרוניות)
HF_MAX_SIZE_MB = 25 

class FreeTranscriber:
    def __init__(self):
        # סדר עדיפות: AssemblyAI (דיאריזציה), HuggingFace (חינם), Google (במקרה הצורך)
        self.providers = ["assemblyai", "huggingface", "google"]
    
    # ------------------------------------------------------------------------
    # פונקציה 1: Hugging Face (חינם) 
    # ------------------------------------------------------------------------
    def transcribe_huggingface(self, audio_file_path) -> Optional[str]:
        """Hugging Face Whisper - חינמי לגמרי. מתאים לקבצים קצרים."""
        try:
            file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
            if file_size_mb > HF_MAX_SIZE_MB:
                st.warning(f"🤖 Hugging Face דורש קבצים קטנים מ-{HF_MAX_SIZE_MB}MB. הקובץ הנוכחי הוא {file_size_mb:.2f}MB. מדלג.")
                return None
            if not st.secrets.get('HF_TOKEN'):
                st.warning("🤖 חסר HF_TOKEN. מדלג על Hugging Face.")
                return None

            API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
            headers = {"Authorization": f"Bearer {st.secrets.get('HF_TOKEN')}"}
            
            with open(audio_file_path, "rb") as f:
                data = f.read()
            
            response = requests.post(API_URL, headers=headers, data=data)
            result = response.json()
            
            if 'text' in result:
                return result['text']
            st.warning(f"🤖 Hugging Face נכשל עם תגובה: {result.get('error', 'לא ידוע')}")
            return None
        except Exception as e:
            st.warning(f"🤖 Hugging Face נכשל: {str(e)}")
            return None

    # ------------------------------------------------------------------------
    # פונקציה 2: AssemblyAI (דיאריזציה וסטטוס) - מתוקן לטיפול בשגיאות Streamlit
    # ------------------------------------------------------------------------
    def transcribe_assemblyai(self, audio_file_path, enable_diarization=False) -> Optional[str]:
        """AssemblyAI - חינמי לניסוי. תומך בדיאריזציה אסינכרונית."""
        try:
            if not st.secrets.get('ASSEMBLYAI_TOKEN'):
                st.error("🛑 חסר ASSEMBLYAI_TOKEN.")
                return None
                
            headers = {
                "authorization": st.secrets.get('ASSEMBLYAI_TOKEN'),
                "content-type": "application/json"
            }
            
            # יצירת פלייס-הולדר פעם אחת לסטטוס
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

            # 2. תחילת תימלול עם הגדרות דיאריזציה
            transcript_url = "https://api.assemblyai.com/v2/transcript"
            
            data = {"audio_url": upload_data["upload_url"], "language_code": "he"} 
            if enable_diarization:
                data["speaker_diarization"] = True
                status_placeholder.info("🗣️ Diarization מופעלת. ממתין לתוצאות...")
                
            transcript_response = requests.post(
                transcript_url,
                headers=headers,
                json=data
            )
            transcript_data = transcript_response.json()
            
            # בדיקת כשל ביצירת משימה
            if 'error' in transcript_data:
                status_placeholder.error(f"❌ AssemblyAI כשל ביצירת משימה: {transcript_data['error']}")
                return None
            if 'id' not in transcript_data:
                status_placeholder.error(f"❌ AssemblyAI כשל: לא התקבל ID משימה.")
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
                    
                    # *** התיקון לשגיאת removeChild: מנקה את ה-placeholder לפני היציאה ***
                    status_placeholder.empty() 
                    
                    # --- עיבוד פלט הפרדת דוברים ---
                    if enable_diarization and 'utterances' in polling_data:
                        formatted_text = ""
                        for utterance in polling_data['utterances']: 
                            formatted_text += f"**דובר {utterance['speaker']}:** {utterance['text']}\n\n"
                        return formatted_text.strip()
                    # ----------------------------------
                    
                    return polling_data["text"] 
                    
                elif current_status == "error":
                    status_placeholder.error(f"❌ AssemblyAI: שגיאה בתימלול: {polling_data.get('error')}")
                    return None
                
                time.sleep(5) 
                
        except Exception as e:
            st.warning(f"🎧 AssemblyAI נכשל: {str(e)}")
            return None

    # ------------------------------------------------------------------------
    # פונקציה 3: Google Speech-to-Text 
    # ------------------------------------------------------------------------
    def transcribe_google(self, audio_file_path) -> Optional[str]:
        """Google Speech-to-Text - חינמי 60 דקות/חודש."""
        try:
            # דורש משתנה סביבה GOOGLE_APPLICATION_CREDENTIALS
            # נדלג על בדיקת המפתח כאן כיוון שגוגל משתמש ב-credentials.json
            client = speech.SpeechClient()
            
            with io.open(audio_file_path, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                language_code="he-IL", 
            )
            
            # בדיקה גסה של גודל קובץ למניעת כשל ב-API סינכרוני
            if len(content) > 10 * 1024 * 1024: 
                 st.warning("☁️ קובץ גדול מדי לתמלול סינכרוני בגוגל. מדלג.")
                 return None

            response = client.recognize(config=config, audio=audio)
            
            text = " ".join([result.alternatives[0].transcript for result in response.results])
            return text.strip() if text else None
            
        except Exception as e:
            st.warning(f"☁️ Google Speech-to-Text נכשל. שגיאה: {str(e)}")
            return None

    # ------------------------------------------------------------------------
    # פונקציה ראשית: Smart Transcribe
    # ------------------------------------------------------------------------
    def smart_transcribe(self, audio_file_path, enable_diarization=False) -> Tuple[str, str]:
        """מנסה את כל ה-APIs לפי סדר עדיפות, ומכבד את בקשת הדיאריזציה."""
        st.info("🔍 מחפש את השירות המתאים ביותר...")
        
        provider_order = self.providers
        
        for provider in provider_order:
            st.write(f"🔄 מנסה {provider}...")
            
            result = None
            if provider == "assemblyai":
                result = self.transcribe_assemblyai(audio_file_path, enable_diarization)
            
            elif enable_diarization:
                st.warning(f"⚠️ {provider} אינו תומך בהפרדת דוברים בקוד זה. מדלג.")
                continue

            elif provider == "huggingface":
                result = self.transcribe_huggingface(audio_file_path)
            
            elif provider == "google":
                result = self.transcribe_google(audio_file_path)
            
            if result:
                st.success(f"✅ הצלחה עם {provider}!")
                return result, provider
        
        return "❌ כל השירותים נכשלו. ודא ש-API Keys נכונים ונסה שוב מאוחר יותר.", "none"

# 🎨 ממשק משתמש (Streamlit Frontend)
def main():
    st.set_page_config(page_title="🎤 פלטפורמת תימלול שמע חכמה", page_icon="🎵")
    
    # --- כותרת מותאמת אישית ---
    st.markdown("""
    ## 🎤 מתמלל חכם נבנה ע"י א קצבורג
    """)
    # -------------------------
    
    uploaded_file = st.file_uploader("העלה קובץ שמע", type=['mp3', 'wav', 'm4a', 'ogg', 'mp4'])
    
    # הוספת תיבת הסימון להפרדת דוברים
    enable_diarization = st.checkbox("🗣️ הפעל הפרדת דוברים (Speaker Diarization)", value=False, 
                                     help="פועל רק בשירות AssemblyAI ומאריך את זמן העיבוד.")
    
    if uploaded_file and st.button("🎯 התחל תימלול חכם"):
        
        if enable_diarization and not st.secrets.get('ASSEMBLYAI_TOKEN'):
             st.error("🛑 לא ניתן לבצע הפרדת דוברים ללא מפתח AssemblyAI שהוגדר ב-.streamlit/secrets.toml")
             return

        # שמירת הקובץ הזמני
        file_extension = uploaded_file.name.split('.')[-1]
        
        temp_dir = 'temp_uploads'
        os.makedirs(temp_dir, exist_ok=True)
        audio_path = os.path.join(temp_dir, f"temp_audio.{file_extension}")

        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        
        try:
            transcriber = FreeTranscriber()
            result, provider = transcriber.smart_transcribe(audio_path, enable_diarization=enable_diarization)
        finally:
            # ניקוי הקובץ הזמני תמיד
            os.unlink(audio_path)
            # הסרת התיקייה אם היא ריקה
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        
        # הצגת התוצאה
        st.subheader("📄 תוצאות התימלול:")
        
        if "❌" in result:
             st.error(result)
        else:
            st.markdown(result) 
            st.info(f"**שותף ששימש לתמלול:** {provider}")
        
            # אפשרות הורדה
            st.download_button(
                label="📥 הורד כקובץ טקסט",
                data=result,
                file_name="transcript.txt",
                mime="text/plain"
            )

# 🚀 הפעלה
if __name__ == "__main__":
    main()