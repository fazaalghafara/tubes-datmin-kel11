import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve, silhouette_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



st.set_page_config(
    page_title="Bank Deposit Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Marketing – Deposit Prediction")



@st.cache_data
def load_data():
    df = pd.read_csv("bank.csv")
    return df

try:
    df = load_data()
    data_loaded_ok = True
except FileNotFoundError:
    st.error("File `bank.csv` tidak ditemukan. Pastikan file ada di direktori yang sama dengan `app.py`.")
    st.stop()



@st.cache_resource
def train_models(df):
    X = df.drop('deposit', axis=1).copy()
    y = df['deposit'].copy()


    categorical_features = X.select_dtypes(include='object').columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)


    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42
    )


    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)


    nb = GaussianNB()
    nb.fit(X_train, y_train)


    cluster_features = ['age', 'balance']
    X_cluster = df[cluster_features]
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster_scaled)
    sil_score = silhouette_score(X_cluster_scaled, cluster_labels)

    return (lr, nb, le_target, X_test, y_test,
            X_encoded.columns.tolist(), kmeans, scaler_cluster,
            X_cluster_scaled, cluster_labels, sil_score)

(lr, nb, le_target, X_test, y_test, feature_cols,
 kmeans, scaler_cluster, X_cluster_scaled, cluster_labels, sil_score) = train_models(df)


menu = st.sidebar.radio(
    "📂 Navigasi",
    ["🏠 Beranda", "📊 Eksplorasi Data", "🧩 Segmentasi Nasabah", "📈 Evaluasi Model", "🔮 Prediksi Baru"]
)



if menu == "🏠 Beranda":
    st.header("🏠 Beranda")
    st.markdown("Prediksi apakah nasabah akan melakukan deposit berdasarkan data kampanye bank.")

    if data_loaded_ok:
        st.success(f"Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

    st.subheader("Tentang Dashboard Ini")
    st.markdown("""
    Dashboard ini dibangun menggunakan **Streamlit** untuk mendukung analisis dan prediksi
    pada data kampanye pemasaran bank (*Bank Marketing Dataset*). Terdapat 4 menu utama
    yang dapat diakses melalui sidebar:

    - **📊 Eksplorasi Data** — statistik deskriptif, distribusi target, dan korelasi antar fitur
    - **🧩 Segmentasi Nasabah** — pengelompokan nasabah berdasarkan usia dan saldo (K-Means)
    - **📈 Evaluasi Model** — perbandingan performa Logistic Regression dan Naive Bayes
    - **🔮 Prediksi Baru** — formulir untuk memprediksi nasabah baru secara langsung
    """)

    st.subheader("Contoh Data")
    st.dataframe(df.head())



elif menu == "📊 Eksplorasi Data":
    st.header("📊 Eksplorasi Data")

    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Target (Deposit)")
        fig, ax = plt.subplots()
        sns.countplot(x='deposit', data=df, ax=ax)
        ax.set_title("Distribusi Deposit")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Missing Values")
        missing = df.isnull().sum().reset_index()
        missing.columns = ['Kolom', 'Jumlah Missing']
        st.dataframe(missing)

    st.subheader("Correlation Matrix")
    numerical_cols = df.select_dtypes(include=np.number).columns
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[numerical_cols].corr(), annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)
    plt.close()

    st.subheader("Duration vs Deposit")
    fig, ax = plt.subplots()
    sns.boxplot(x='deposit', y='duration', data=df, ax=ax)
    ax.set_title("Duration vs Deposit")
    st.pyplot(fig)
    plt.close()



elif menu == "🧩 Segmentasi Nasabah":
    st.header("🧩 Segmentasi Nasabah (K-Means, k=4)")
    st.markdown(f"Segmentasi menggunakan fitur **age** dan **balance**. Silhouette Score: **{sil_score:.3f}**")

    st.subheader("Visualisasi Cluster")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_cluster_scaled[:, 0], X_cluster_scaled[:, 1],
        c=cluster_labels, cmap='viridis', alpha=0.6
    )
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200,
               edgecolor='black', label='Centroid')
    ax.set_xlabel('Age (scaled)')
    ax.set_ylabel('Balance (scaled)')
    ax.set_title('Visualisasi Cluster K-Means')
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.subheader("Profil Tiap Cluster")
    df_cluster_profile = df[['age', 'balance']].copy()
    df_cluster_profile['cluster'] = cluster_labels
    st.dataframe(df_cluster_profile.groupby('cluster').agg(['mean', 'median', 'count']))



elif menu == "📈 Evaluasi Model":
    st.header("📈 Evaluasi Model")

    y_pred_lr = lr.predict(X_test)
    y_pred_nb = nb.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    y_prob_nb = nb.predict_proba(X_test)[:, 1]

    st.subheader("Perbandingan Metrik Model")
    hasil_model = pd.DataFrame({
        'Model': ['Logistic Regression', 'Naive Bayes'],
        'Accuracy':  [accuracy_score(y_test, y_pred_lr),  accuracy_score(y_test, y_pred_nb)],
        'Precision': [precision_score(y_test, y_pred_lr), precision_score(y_test, y_pred_nb)],
        'Recall':    [recall_score(y_test, y_pred_lr),    recall_score(y_test, y_pred_nb)],
        'F1 Score':  [f1_score(y_test, y_pred_lr),        f1_score(y_test, y_pred_nb)],
        'ROC-AUC':   [roc_auc_score(y_test, y_prob_lr),   roc_auc_score(y_test, y_prob_nb)],
    })
    st.dataframe(hasil_model.set_index('Model').style.highlight_max(axis=0, color='#d4edda'))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix – Logistic Regression")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Confusion Matrix – Naive Bayes")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', ax=ax, cmap='Oranges')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close()

    st.subheader("Perbandingan ROC Curve")
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={roc_auc_score(y_test, y_prob_lr):.3f})")
    ax.plot(fpr_nb, tpr_nb, label=f"Naive Bayes (AUC={roc_auc_score(y_test, y_prob_nb):.3f})")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Perbandingan ROC Curve")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.subheader("Faktor Paling Berpengaruh (Logistic Regression)")
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'coefficient': lr.coef_[0]
    })
    feature_importance['abs_coefficient'] = feature_importance['coefficient'].abs()
    feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1D9E75' if c > 0 else '#D85A30' for c in feature_importance['coefficient']]
    ax.barh(feature_importance['feature'], feature_importance['coefficient'], color=colors)
    ax.set_xlabel('Koefisien Logistic Regression')
    ax.set_title('15 Faktor Paling Berpengaruh terhadap Keputusan Deposit')
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close()


elif menu == "🔮 Prediksi Baru":
    st.header("🔮 Prediksi Nasabah Baru")
    st.markdown("Isi data nasabah di bawah ini untuk memprediksi kemungkinan deposit.")

    st.subheader("Input Data Nasabah")
    col1, col2, col3 = st.columns(3)

    with col1:
        age      = st.number_input("Age", min_value=18, max_value=100, value=35)
        job      = st.selectbox("Job", df['job'].unique())
        marital  = st.selectbox("Marital", df['marital'].unique())
        education= st.selectbox("Education", df['education'].unique())
        default  = st.selectbox("Default", df['default'].unique())

    with col2:
        balance  = st.number_input("Balance", value=1000)
        housing  = st.selectbox("Housing", df['housing'].unique())
        loan     = st.selectbox("Loan", df['loan'].unique())
        contact  = st.selectbox("Contact", df['contact'].unique())
        day      = st.number_input("Day", min_value=1, max_value=31, value=15)

    with col3:
        month    = st.selectbox("Month", df['month'].unique())
        duration = st.number_input("Duration (detik)", min_value=0, value=200)
        campaign = st.number_input("Campaign", min_value=1, value=1)
        pdays    = st.number_input("Pdays", min_value=-1, value=-1)
        previous = st.number_input("Previous", min_value=0, value=0)
        poutcome = st.selectbox("Poutcome", df['poutcome'].unique())

    st.subheader("Pilih Model")
    model_choice = st.radio(
        "Model yang digunakan:",
        ["Logistic Regression", "Naive Bayes"],
        horizontal=True
    )

    if st.button("🔮 Prediksi Sekarang", type="primary"):


        numeric_input = {
            'age': age, 'balance': balance, 'day': day,
            'duration': duration, 'campaign': campaign,
            'pdays': pdays, 'previous': previous
        }
        categorical_input = {
            'job': job, 'marital': marital, 'education': education,
            'default': default, 'housing': housing, 'loan': loan,
            'contact': contact, 'month': month, 'poutcome': poutcome
        }

.
        input_final = pd.DataFrame(0, index=[0], columns=feature_cols)


        for col, val in numeric_input.items():
            if col in input_final.columns:
                input_final[col] = val

        for col, val in categorical_input.items():
            dummy_col = f"{col}_{val}"
            if dummy_col in input_final.columns:
                input_final[dummy_col] = 1



        model = lr if model_choice == "Logistic Regression" else nb
        pred = model.predict(input_final)[0]
        prob = model.predict_proba(input_final)[0]

        label = le_target.inverse_transform([pred])[0]

        st.subheader("Hasil Prediksi")
        if label == 'yes':
            st.success(f"✅ Nasabah **AKAN** melakukan deposit  (Probabilitas: {prob[pred]*100:.1f}%)")
        else:
            st.error(f"❌ Nasabah **TIDAK AKAN** melakukan deposit  (Probabilitas: {prob[pred]*100:.1f}%)")

        prob_df = pd.DataFrame({
            'Kelas': le_target.classes_,
            'Probabilitas': prob
        })
        fig, ax = plt.subplots(figsize=(5, 2))
        sns.barplot(x='Kelas', y='Probabilitas', data=prob_df, ax=ax, palette='Blues_d')
        ax.set_ylim(0, 1)
        ax.set_title("Distribusi Probabilitas")
        st.pyplot(fig)
        plt.close()
