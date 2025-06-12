import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------- SETUP ----------
st.set_page_config(page_title="Dropout Predictor", layout="wide")
st.title("🎓 Prediksi Risiko Dropout Mahasiswa")

# ---------- LOAD & PREPROCESS DATA ----------
@st.cache_data

def load_data():
    df = pd.read_csv("data.csv", delimiter=';')
    columns_to_drop = [
        'Unemployment_rate', 'Inflation_rate', 'GDP',
        'Curricular_units_1st_sem_credited',
        'Curricular_units_2nd_sem_credited',
        'Curricular_units_1st_sem_without_evaluations',
        'Curricular_units_2nd_sem_without_evaluations'
    ]
    df.drop(columns=columns_to_drop, inplace=True)
    return df

df = load_data()

# Encode target
le = LabelEncoder()
df['Status'] = le.fit_transform(df['Status'])  # Dropout=0, Enrolled=1, Graduate=2

# Fitur numerik dan scaler
numeric_columns = [
    'Application_order', 'Previous_qualification_grade', 'Admission_grade',
    'Age_at_enrollment', 'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade'
]
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# One-hot encoding
categorical_columns = ['Gender', 'Scholarship_holder', 'Marital_status', 'Daytime_evening_attendance']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# ---------- MODEL ----------
X = df.drop('Status', axis=1)
y = df['Status']
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# ---------- INPUT FORM ----------
st.sidebar.header("Input Data Mahasiswa")
def user_input():
    input_dict = {}
    input_dict['Application_order'] = st.sidebar.slider("Urutan Pilihan Pendaftaran", 0, 9, 1)
    input_dict['Previous_qualification_grade'] = st.sidebar.slider("Nilai Pendidikan Sebelumnya", 0, 200, 120)
    input_dict['Admission_grade'] = st.sidebar.slider("Nilai Saat Masuk", 0, 200, 120)
    input_dict['Age_at_enrollment'] = st.sidebar.slider("Usia Saat Mendaftar", 15, 50, 20)
    input_dict['Curricular_units_1st_sem_enrolled'] = st.sidebar.slider("Mata Kuliah Diambil Smt 1", 0, 10, 6)
    input_dict['Curricular_units_1st_sem_evaluations'] = st.sidebar.slider("Dievaluasi Smt 1", 0, 10, 6)
    input_dict['Curricular_units_1st_sem_approved'] = st.sidebar.slider("Lulus Smt 1", 0, 10, 5)
    input_dict['Curricular_units_1st_sem_grade'] = st.sidebar.slider("Nilai Rata-rata Smt 1", 0.0, 20.0, 12.0)
    input_dict['Curricular_units_2nd_sem_enrolled'] = st.sidebar.slider("Mata Kuliah Diambil Smt 2", 0, 10, 6)
    input_dict['Curricular_units_2nd_sem_evaluations'] = st.sidebar.slider("Dievaluasi Smt 2", 0, 10, 6)
    input_dict['Curricular_units_2nd_sem_approved'] = st.sidebar.slider("Lulus Smt 2", 0, 10, 5)
    input_dict['Curricular_units_2nd_sem_grade'] = st.sidebar.slider("Nilai Rata-rata Smt 2", 0.0, 20.0, 12.0)
    input_dict['Tuition_fees_up_to_date'] = st.sidebar.selectbox("Biaya Kuliah Lunas?", [0, 1])
    input_dict['Displaced'] = st.sidebar.selectbox("Mahasiswa Merantau?", [0, 1])
    input_dict['Scholarship_holder'] = st.sidebar.selectbox("Penerima Beasiswa?", [0, 1])
    input_dict['Gender'] = st.sidebar.selectbox("Jenis Kelamin", ['male', 'female'])
    input_dict['Marital_status'] = st.sidebar.selectbox("Status Pernikahan", [1, 2, 3, 4, 5, 6])
    input_dict['Daytime_evening_attendance'] = st.sidebar.selectbox("Kuliah Pagi atau Malam?", [0, 1])
    return input_dict

user_data = user_input()

# ---------- PROSES PREDIKSI ----------
user_df = pd.DataFrame([user_data])

# One-hot encoding manual agar cocok dengan model
user_df['Gender_male'] = 1 if user_data['Gender'] == 'male' else 0
user_df.drop('Gender', axis=1, inplace=True)

for col in ['Scholarship_holder', 'Marital_status', 'Daytime_evening_attendance']:
    for option in df.columns:
        if option.startswith(col + "_"):
            value = int(option.split('_')[-1])
            user_df[option] = 1 if user_data[col] == value else 0
    user_df.drop(col, axis=1, inplace=True)

# Isi kolom yang tidak ada di inputan dengan 0 (agar shape cocok)
for col in X.columns:
    if col not in user_df.columns:
        user_df[col] = 0

# Susun kolom agar urutan cocok
user_df = user_df[X.columns]

# Normalisasi fitur numerik
user_df[numeric_columns] = scaler.transform(user_df[numeric_columns])

# Prediksi dan Probabilitas
prediction = model.predict(user_df)[0]
pred_proba = model.predict_proba(user_df).max()
status_label = le.inverse_transform([prediction])[0]

# ---------- OUTPUT ----------
st.subheader("📊 Hasil Prediksi")
st.write(f"**Status Prediksi:** {status_label} (Probabilitas: {pred_proba:.2f})")

# ---------- REKOMENDASI ----------
rekomendasi = []
if user_df['Curricular_units_1st_sem_approved'].values[0] < 0:
    rekomendasi.append("- Tawarkan bimbingan akademik tambahan.")
if user_df['Admission_grade'].values[0] < 0:
    rekomendasi.append("- Berikan pelatihan belajar mandiri atau workshop persiapan kuliah.")
if user_df['Tuition_fees_up_to_date'].values[0] == 0:
    rekomendasi.append("- Hubungi siswa untuk menyelesaikan administrasi keuangan.")
if user_df['Scholarship_holder_1'].values[0] == 0:
    rekomendasi.append("- Tawarkan bantuan finansial atau beasiswa tambahan.")
if user_df['Displaced'].values[0] == 1:
    rekomendasi.append("- Sediakan layanan dukungan untuk siswa yang tinggal jauh dari rumah.")

if prediction == 0:  # Dropout
    st.subheader("🛠️ Rekomendasi Tindakan")
    if rekomendasi:
        for r in rekomendasi:
            st.write(r)
    else:
        st.write("- Siswa berisiko dropout, tapi tidak ada faktor dominan yang terdeteksi.")
else:
    st.success("Siswa tidak termasuk dalam risiko dropout tinggi.")