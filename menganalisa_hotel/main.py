import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#0. MEMBUAT DATASET DUMMY (RESERVASI HOTEL)

# membuat data observasi dengan fitur numerik, kategori, nilai hilang, dan outlier
data = {
    'id_reservasi': range(1, 101),
    'tipe_kamar': np.random.choice(['Standard', 'Deluxe', 'Suite'], 100),
    'lama_menginap_hari': np.random.randint(1, 7, 100).astype(float),
    'total_tagihan_rp': np.random.randint(300000, 2000000, 100).astype(float)
}
df_hotel = pd.DataFrame(data)

# membuat Missing Values (Nilai Kosong)
df_hotel.loc[10:15, 'lama_menginap_hari'] = np.nan 

#  Outlier (Data Ekstrem)
df_hotel.loc[25, 'total_tagihan_rp'] = 15000000.0  
df_hotel.loc[50, 'total_tagihan_rp'] = 20000000.0  

# Menyimpan ke dalam  CSV agar saya berlatih  membaca data eksternal
df_hotel.to_csv('data_reservasi_hotel.csv', index=False)

print("--- Memulai Praktik Exploratory Data Analysis & Data Cleaning ---\n")


# 1. EKSPLORASI DATA DASAR (EDA)

# Tujuan: Mengetahui karakteristik data dan statistik ringkas
df = pd.read_csv('data_reservasi_hotel.csv')

print("1. Statistik Ringkas Data:")
print(df.describe())

# Visualisasi distribusi tagihan sebelum dibersihkan
plt.figure(figsize=(8, 4))
sns.histplot(df['total_tagihan_rp'], bins=20, kde=True)
plt.title("Distribusi Total Tagihan (Terdapat Outlier)")
plt.show()


# 2. SAMPLING DATA

# Mencoba Mengambil 5 baris acak untuk memeriksa kelayakan isi dari data
print("\n2. Sampel Data Acak (5 Baris):")
sample = df.sample(n=5, replace=False)
print(sample)


# 3. MENANGANI NILAI HILANG (MISSING VALUES)
print("\n3. Pengecekan Nilai Hilang:")
print(df.isnull().sum()) # Cek nilai hilang[cite: 2]

# Mengimputasi nilai hilang pada 'lama_menginap_hari' dengan median agar tidak bias
median_menginap = df['lama_menginap_hari'].median()
df['lama_menginap_hari'].fillna(median_menginap, inplace=True)

print(f"-> Nilai kosong pada 'lama_menginap_hari' telah diisi dengan median: {median_menginap}")


# 4. DETEKSI & PENANGANAN OUTLIER DENGAN IQR
# Menghitung batas outlier menggunakan interquartile range (IQR)
Q1 = df['total_tagihan_rp'].quantile(0.25)
Q3 = df['total_tagihan_rp'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Mendeteksi outlier pada tagihan[cite: 1, 2]
outliers = df[(df['total_tagihan_rp'] < lower_bound) | (df['total_tagihan_rp'] > upper_bound)]
print(f"\n4. Ditemukan {len(outliers)} data outlier ekstrem. Data ini akan dihapus.")

# Menghapus baris outlier agar model tidak salah memprediksi
df = df[~df.index.isin(outliers.index)]


# 5. FEATURE ENGINEERING (REKAYASA FITUR)
# Membuat fitur baru: Estimasi biaya per malam
df['biaya_per_malam'] = df['total_tagihan_rp'] / df['lama_menginap_hari']

# Mengubah fitur kategori (tipe_kamar) menjadi format numerik agar dipahami model
df = pd.get_dummies(df, columns=['tipe_kamar'])
print("\n5. Feature Engineering Berhasil. Kolom baru telah ditambahkan.")


# 6. FEATURE SCALING (TRANSFORMASI FITUR)
# Membuat fitur berskala sama agar model stabil
scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

# Standardisasi (Z-score) pada fitur 'biaya_per_malam'
df['scaled_biaya_per_malam'] = scaler.fit_transform(df[['biaya_per_malam']])

# Normalisasi (Min-Max 0-1) pada 'total_tagihan_rp'
df['normalized_tagihan'] = minmax_scaler.fit_transform(df[['total_tagihan_rp']])

print("\n6. Hasil Akhir Data Setelah Dibersihkan & Diskalakan (5 baris pertama):")
print(df.head())

# Visualisasi Boxplot setelah bersih
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['normalized_tagihan'])
plt.title("Boxplot Tagihan yang Telah Dinormalisasi (Tanpa Outlier)")
plt.show()