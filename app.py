import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

st.title("Prediksi Dropout Mahasiswa & Rekomendasi Tindakan")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/data.csv", delimiter=';')
    drop_cols = ['Unemployment_rate', 'Inflation_rate', 'GDP',
                 'Curricular_units_1st_sem_credited', 'Curricular_units_2nd_sem_credited',
                 'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_without_evaluations']
    df = df.drop(columns=drop_cols)
    df['Status'] = LabelEncoder().fit_transform(df['Status'])
    return df

df = load_data()

# Preprocessing
numeric_cols = ['Application_order', 'Previous_qualification_grade', 'Admission_grade',
                'Age_at_enrollment', 'Curricular_units_1st_sem_enrolled',
                'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
                'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_enrolled',
                'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
                'Curricular_units_2nd_sem_grade']
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df = pd.get_dummies(df, columns=['Gender', 'Scholarship_holder', 'Marital_status', 'Daytime_evening_attendance'], drop_first=True)

# Split
X = df.drop('Status', axis=1)
y = df['Status']
model = RandomForestClassifier(random_state=42).fit(X, y)

# Input user
st.sidebar.header("Masukkan Data Siswa")
input_data = {}
for col in X.columns:
    val = st.sidebar.number_input(f"{col}", value=0.0)
    input_data[col] = val

# Prediksi
user_df = pd.DataFrame([input_data])
pred = model.predict(user_df)[0]
status_map = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
st.subheader(f"📌 Prediksi Status: **{status_map[pred]}**")

# Rekomendasi
st.subheader("🛠 Rekomendasi Tindakan")
rekomendasi = []
if user_df['Curricular_units_1st_sem_approved'][0] < 0:
    rekomendasi.append("- Tawarkan bimbingan akademik tambahan.")
if user_df['Admission_grade'][0] < 0:
    rekomendasi.append("- Workshop belajar mandiri.")
if 'Tuition_fees_up_to_date' in user_df.columns and user_df['Tuition_fees_up_to_date'][0] == 0:
    rekomendasi.append("- Hubungi untuk perbaikan administrasi.")
if rekomendasi:
    for r in rekomendasi:
        st.write(r)
else:
    st.write("✅ Tidak ada rekomendasi khusus.")
