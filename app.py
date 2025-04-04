import streamlit as st
import numpy as np
import joblib
from pyngrok import ngrok

# فتح نفق على البورت الذي يعمل عليه Streamlit (عادة 8501)
public_url = ngrok.connect(8501)
print(f" * Streamlit app is live at: {public_url}")

# Load the trained model and scaler
model = joblib.load("alzheimerr_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page Configuration
st.set_page_config(page_title="Alzheimer's Prediction", layout="centered")

# Title with Brain Emoji
st.markdown("<h1 style='text-align: center; color: #6A0DAD;'>🧠 Alzheimer's Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>أدخل بياناتك للحصول على التنبؤ</h4>", unsafe_allow_html=True)
st.write("---")

# User Input Form
with st.form(key="alzheimers_form"):
    # Age input first
    age = st.number_input("العمر (بالسنوات):", min_value=40, max_value=100, step=1)

    # Gender selection
    gender = st.selectbox("الجنس:", ["ذكر", "أنثى"])

    # Lifestyle & Health Factors
    st.write("### 🏥 العوامل الصحية و أسلوب الحياة")
    smoking = st.selectbox("هل تدخن؟", ["نعم", "لا"])
    diabetes = st.selectbox("هل لديك مرض السكري؟", ["نعم", "لا"])
    cholesterol_hdl = st.number_input("مستوى HDL (المفيد) للكوليسترول:", min_value=20.0, max_value=100.0, step=0.1)
    systolic_bp = st.number_input("ضغط الدم الانقباضي (طبيعي بين 90-120):", min_value=80, max_value=200, step=1)
    bmi = st.number_input("مؤشر كتلة الجسم (BMI) (18.5 - 24.9 مثالي):", min_value=10.0, max_value=50.0, step=0.1)

    # Cognitive & Behavioral Factors
    st.write("### 🧠 الوظائف الإدراكية و السلوك")
    memory_complaints = st.selectbox("هل تعاني من مشاكل في الذاكرة؟", ["لا", "نعم"])
    disorientation = st.selectbox("هل تعاني من الارتباك؟", ["لا", "نعم"])
    personality_changes = st.selectbox("هل تعاني من تغييرات في الشخصية؟", ["لا", "نعم"])
    behavioral_problems = st.selectbox("هل لديك مشاكل سلوكية؟", ["لا", "نعم"])

    # Lifestyle & Mental Factors
    st.write("### 🌙 جودة الحياة و النشاط")
    sleep_quality = st.slider("جودة النوم (0 = سيئ جدًا، 10 = ممتاز)", 0, 10, 5)
    physical_activity = st.slider("النشاط البدني (0 = لا شيء، 10 = نشيط جدًا)", 0, 10, 5)
    functional_assessment = st.slider("تقييم الوظائف الإدراكية (0 = ضعيف جدًا، 10 = ممتاز)", 0, 10, 5)
    adl = st.slider("القدرة على أداء الأنشطة اليومية (0 = لا يستطيع، 10 = طبيعي)", 0, 10, 5)
    mmse = st.slider("اختبار الحالة الذهنية (0 = شديد الضعف، 30 = طبيعي)", 0, 30, 15)

    # Submit button
    submit_button = st.form_submit_button(label="🔍 تحليل النتيجة")

# Convert inputs to numerical values
input_data = np.array([
    1 if gender == "ذكر" else 0,
    1 if smoking == "نعم" else 0,
    1 if disorientation == "نعم" else 0,
    1 if diabetes == "نعم" else 0,
    bmi, age, systolic_bp, sleep_quality,
    physical_activity, cholesterol_hdl,
    1 if memory_complaints == "نعم" else 0,
    1 if behavioral_problems == "نعم" else 0,
    functional_assessment, adl, mmse,
    1 if personality_changes == "نعم" else 0
]).reshape(1, -1)

# Normalize input data
input_data = scaler.transform(input_data)

# Predict and display result
if submit_button:
    prediction = model.predict(input_data)[0]
    result_text = "🛑 **قد تكون مصابًا بمرض الزهايمر. يُرجى استشارة الطبيب.**" if prediction == 1 else "✅ **لا يوجد مؤشر واضح للإصابة بالزهايمر.**"
    st.markdown(f"<h3 style='text-align: center; color: #D32F2F;'>{result_text}</h3>", unsafe_allow_html=True)
