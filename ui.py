import streamlit as st
import requests
from PIL import Image

# כותרת האפליקציה
st.title("🧠 מזהה התמונות החכם שלי")
st.write("העלה תמונה, והמודל ינסה לנחש מה יש בה!")

# 1. רכיב להעלאת קבצים
uploaded_file = st.file_uploader("בחר תמונה...", type=["jpg", "jpeg", "png"])

# 2. לוגיקה - מה קורה כשיש תמונה
if uploaded_file is not None:
    # הצגת התמונה שהועלתה
    image = Image.open(uploaded_file)
    st.image(image, caption='התמונה שבחרת', use_column_width=True)
    
    # כפתור לפעולה
    if st.button('🔍 זהה את התמונה!'):
        st.write("שולח לניתוח...")
        
        try:
            # הכנת הקובץ לשליחה
            # (אנחנו צריכים "לאפס" את הקובץ לתחילתו כדי לקרוא ממנו שוב)
            uploaded_file.seek(0)
            files = {'image': uploaded_file.getvalue()}
            
            # שליחת בקשה ל-API המקומי שלנו (שבתוך הדוקר)
            response = requests.post("http://localhost:5000/predict", files=files)
            
            # קבלת התשובה
            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
                count = result['total_seen_this_class']
                
                # הצגת התוצאה בגדול
                st.success(f"זהו **{prediction}**! 🎉")
                st.info(f"עד עכשיו ראיתי {count} תמונות מסוג {prediction}.")
            else:
                st.error("משהו השתבש בשרת...")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"שגיאה בהתחברות לשרת: {e}")
            st.warning("האם Docker Compose רץ? (נסה בטרמינל: curl localhost:5000/predict)")