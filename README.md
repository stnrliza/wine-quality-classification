# Wine Quality Classification Project

## 1. Domain Proyek
Industri anggur global sangat kompetitif, di mana kualitas produk menjadi faktor utama dalam menentukan kepuasan konsumen dan keberhasilan bisnis. Kualitas anggur yang tinggi tidak hanya memperkuat citra merek, tetapi juga meningkatkan nilai jual di pasar. Oleh karena itu, evaluasi kualitas anggur secara akurat dan konsisten sangat penting dalam proses produksi.

Penilaian kualitas anggur secara tradisional biasanya dilakukan secara sensorik oleh panel ahli. Meski akurat, metode ini memiliki beberapa kendala, seperti sifatnya yang subjektif, waktu proses yang lama, biaya tinggi, dan keterbatasan sumber daya manusia. Hal ini menyulitkan penerapan dalam skala besar dan berpotensi menyebabkan ketidakkonsistenan evaluasi yang merugikan reputasi dan pendapatan produsen.

Kondisi tersebut menunjukkan kebutuhan akan metode penilaian kualitas anggur yang lebih objektif, efisien, dan andal. Salah satu pendekatan yang menjanjikan adalah penerapan teknologi machine learning untuk mengembangkan sistem prediksi kualitas berbasis data.

Dengan memanfaatkan data fisikokimia terukur, seperti tingkat keasaman, kadar alkohol, dan nilai pH, machine learning dapat membangun model klasifikasi yang mampu mengidentifikasi kualitas anggur secara otomatis. Model ini diharapkan dapat membantu produsen dalam:

- Mengoptimalkan proses produksi dengan mengidentifikasi faktor utama yang memengaruhi kualitas.
- Menjaga konsistensi produk dengan memastikan standar kualitas sebelum dipasarkan.
- Mendukung pengambilan keputusan strategis, seperti penentuan harga, seleksi bahan baku, atau modifikasi proses fermentasi.

Dengan demikian, produsen anggur dapat mengurangi ketergantungan pada evaluasi manual yang mahal dan variatif, dan beralih ke sistem prediksi yang cepat, objektif, dan akurat berbasis data.

**Referensi:**

[1] Aich, S., Al-Absi, A. A., Hui, K. L., Lee, J. T., & Sain, M. (2018). A classification approach with different feature sets to predict the quality of different types of wine using machine learning techniques. *International Conference on Advanced Communication Technology (ICACT)* (pp. 1–2). Chuncheon, Korea (South). https://doi.org/10.23919/ICACT.2018.8323673

[2] Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modelling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547–553. https://doi.org/10.1016/j.dss.2009.05.016

## 2. Business Understanding
### 2.1. Problem Statements
1. Bagaimana cara memprediksi kualitas anggur merah secara akurat, dikategorikan sebagai Baik atau Buruk, berdasarkan atribut fisikokimia yang terukur?
2. Fitur fisikokimia mana saja yang memiliki pengaruh paling signifikan terhadap kualitas anggur merah?
3. Sejauh mana performa model klasifikasi machine learning dalam memberikan prediksi yang konsisten dan dapat diandalkan untuk mendukung proses produksi anggur?

### 2.2. Goals
1. Mengembangkan model klasifikasi machine learning yang mampu memprediksi kualitas anggur merah berdasarkan data fisikokimia secara akurat.
2. Mengidentifikasi dan menganalisis fitur-fitur fisikokimia yang paling berkontribusi terhadap penentuan kualitas anggur.
3. Mencapai tingkat akurasi prediksi minimal 75% pada data pengujian guna memastikan keandalan model dalam aplikasi nyata.

### 2.3. Solution Statements
1. **Penggunaan Multiple Algorithms**:
Proyek ini akan menerapkan beberapa algoritma klasifikasi, yaitu `Logistic Regression, Random Forest Classifier, dan XGBoost Classifier`, untuk membandingkan performa masing-masing model dan memilih yang terbaik. Untuk menjawab pertanyaan mengenai fitur fisikokimia yang paling berpengaruh, akan dilakukan analisis feature importance pada model berbasis pohon (seperti `Random Forest dan XGBoost`).
2. **Hyperparameter Tuning**:
Pada model yang berpotensi memberikan performa tinggi, seperti `Random Forest dan XGBoost`, akan dilakukan hyperparameter tuning menggunakan GridSearchCV untuk mengoptimalkan hasil prediksi dan melampaui performa baseline.
3. **Metrik Evaluasi Akurasi**:
Akurasi (accuracy_score) akan digunakan sebagai metrik utama untuk menilai kinerja model. Selain itu, Confusion Matrix dan Classification Report akan diaplikasikan untuk mendapatkan analisis lebih mendalam terkait nilai precision, recall, dan F1-score pada masing-masing kelas.


## 3. Data Understanding

Dataset yang digunakan berasal dari "Wine Quality Red" pada UCI Machine Learning Repository. Dataset ini berisi data fisikokimia anggur merah asal Portugal yang dikaitkan dengan penilaian sensorik kualitas anggur.

- **Jumlah Data:** 1599 baris  
- **Jumlah Fitur:** 12 kolom  
- **Kondisi Data:**

| Proses Pemeriksaan       | Deskripsi                                                                       |
|-------------------------|---------------------------------------------------------------------------------|
| Missing Values          | Tidak ditemukan nilai yang kosong dalam dataset                                |
| Duplikasi Data          | Terdapat 240 baris duplikat yang sudah dihapus menggunakan `df.drop_duplicates()` |
| Outlier                 | Ditemukan outlier di sebagian besar fitur menggunakan metode Interquartile Range (IQR) |
| Penanganan Outlier      | Dilakukan capping (winsorizing) untuk meminimalisir pengaruh nilai ekstrim     |

**Sumber Dataset:** [Wine Quality Dataset - UCI ML Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)

---

Setelah mengidentifikasi outlier, penting untuk menangani nilai-nilai ekstrem tersebut agar model tidak bias terhadap data yang tidak representatif. Salah satu pendekatan yang umum adalah **capping** atau **winsorizing**, yaitu mengganti nilai outlier dengan batas bawah atau atas yang ditentukan berdasarkan IQR.

Prinsipnya:  
- Jika nilai lebih kecil dari batas bawah, maka diganti dengan nilai batas bawah.  
- Jika nilai lebih besar dari batas atas, maka diganti dengan nilai batas atas.  

Cara ini menjaga integritas data tanpa menghapus baris, sekaligus mengurangi dampak negatif dari nilai ekstrem.

Contoh kode Python untuk capping:

```python
df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
```

**Variabel/Fitur pada Data:**

| Fitur                  | Keterangan                                                                                                      |
| ---------------------- | --------------------------------------------------------------------------------------------------------------- |
| `fixed acidity`        | Kandungan asam non-volatil yang dominan dan memengaruhi kualitas anggur.                                        |
| `volatile acidity`     | Jumlah asam asetat yang tinggi dapat menyebabkan rasa cuka yang tidak diinginkan.                               |
| `citric acid`          | Memberikan sensasi kesegaran dan rasa khas pada anggur.                                                         |
| `residual sugar`       | Gula yang tersisa setelah fermentasi selesai, memengaruhi tingkat kemanisan anggur.                             |
| `chlorides`            | Kandungan garam dalam anggur.                                                                                   |
| `free sulfur dioxide`  | SO2 bebas yang berfungsi mencegah pertumbuhan mikroorganisme dan oksidasi.                                      |
| `total sulfur dioxide` | Total SO2, termasuk yang bebas dan yang terikat.                                                                |
| `density`              | Kepadatan cairan anggur.                                                                                        |
| `pH`                   | Ukuran tingkat keasaman atau kebasaan anggur.                                                                   |
| `sulphates`            | Senyawa sulfur dioksida yang berperan sebagai antimikroba dan antioksidan.                                      |
| `alcohol`              | Persentase kadar alkohol dalam anggur.                                                                          |
| `quality`              | Skor kualitas anggur berdasarkan penilaian sensorik, skala 0 sampai 10. Fitur target asli yang akan diprediksi. |


## 4. Data Preparation
Tahap ini bertujuan menyiapkan data agar siap digunakan untuk pemodelan machine learning.

1. **Penghapusan Duplikasi Data**  
   - Dataset diperiksa dan ditemukan 240 baris duplikat.  
   - Duplikasi data dihapus menggunakan fungsi `df.drop_duplicates()`, memastikan data lebih bersih dan tidak bias.

2. **Penanganan Outlier**  
   - Outlier terdeteksi pada sebagian besar fitur dengan metode Interquartile Range (IQR).  
   - Penanganan outlier dilakukan menggunakan capping (winsorizing) dengan fungsi `cap_outliers_iqr`, mengganti nilai ekstrim dengan nilai batas atas/bawah yang sesuai, tanpa menghapus baris data.

3. **Mendefinisikan Masalah Klasifikasi Biner**  
   - Kolom `quality` yang awalnya berskala 3–8 diubah menjadi klasifikasi biner:  
     Anggur dengan `quality` < 6 dikategorikan sebagai **'Buruk' (0)**.  
     Anggur dengan `quality` ≥ 6 dikategorikan sebagai **'Baik' (1)**.  
     Dibuat kolom baru `quality_label`.

4. **Pemisahan Fitur (X) dan Target (y)**  
   - **Fitur (X)**: Semua kolom fisikokimia, kecuali `quality` dan `quality_label`.  
   - **Target (y)**: Kolom `quality_label`.

5. **Pemisahan Data Training dan Testing**  
   - Menggunakan `train_test_split` dengan proporsi 80:20.  
   - `random_state=42` untuk hasil yang konsisten dan `stratify=y` untuk keseimbangan kelas.

6. **Feature Scaling (StandardScaler)**  
   - Standardisasi data dilakukan menggunakan `StandardScaler`, mengubah distribusi fitur menjadi rata-rata 0 dan standar deviasi 1.  
   - Scaling ini penting untuk model yang sensitif terhadap skala fitur.


## 5. Model Deployment

Pada tahap ini, tiga model klasifikasi diimplementasikan, dilatih, dan dioptimalkan untuk memprediksi kualitas anggur. Pemilihan model didasarkan pada kemampuannya dalam menangani masalah klasifikasi biner dan mengidentifikasi pola yang tersembunyi dalam data.

Untuk menjaga konsistensi dalam proses pelatihan dan evaluasi, digunakan sebuah fungsi kustom bernama `train_and_evaluate_model`. Fungsi ini menerima input berupa model dan data pelatihan/pengujian, kemudian:
- Melatih model dengan data pelatihan
- Melakukan prediksi pada data pengujian
- Mencetak metrik evaluasi seperti `accuracy_score`, `confusion_matrix`, dan `classification_report`
- Menampilkan visualisasi confusion matrix untuk mempermudah analisis performa model
### 5.1 Model 1: Logistic Regression
**1. Pembahasan Cara Kerja**  
Logistic Regression merupakan algoritma klasifikasi linier yang digunakan untuk memodelkan probabilitas suatu kelas dalam masalah klasifikasi biner. Algoritma ini menggunakan fungsi sigmoid untuk mengkonversi kombinasi linier dari fitur input menjadi probabilitas bernilai antara 0 dan 1. Jika nilai probabilitas melebihi ambang batas tertentu (misalnya 0.5), maka data diklasifikasikan ke kelas positif, sebaliknya ke kelas negatif. Model ini bekerja dengan mencari hubungan linier antara fitur dan log-odds dari target.

**2. Pembahasan Parameter**  
Model ini dilatih menggunakan pustaka Scikit-learn dengan parameter sebagai berikut:
- `random_state=42`: Digunakan untuk memastikan hasil yang konsisten saat pelatihan diulang.
- `solver='liblinear'`: Digunakan karena efisien untuk dataset kecil dan mendukung regularisasi L1 maupun L2.

Parameter lainnya dibiarkan sebagai default karena telah sesuai dengan karakteristik data.

**3. Kelebihan/Kekurangan**  
- **Kelebihan:** Mudah diinterpretasikan, efisien dan cepat dilatih, serta menghasilkan probabilitas yang dapat digunakan untuk analisis lebih lanjut.  
- **Kekurangan:** Asumsi hubungan linier dapat membatasi performa ketika data memiliki pola non-linier yang kompleks. Rentan terhadap outlier dan memerlukan penskalaan fitur.

### 5.2 Model 2: Random Forest Classifier (Tuned)
**1. Pembahasan Cara Kerja**  
Random Forest adalah algoritma ensambel berbasis pohon keputusan yang terdiri dari banyak pohon yang dilatih secara independen. Setiap pohon dilatih menggunakan teknik bootstrapping (pengambilan sampel acak dengan penggantian) dan subset acak dari fitur. Prediksi akhir diambil berdasarkan voting mayoritas dari seluruh pohon. Pendekatan ini mengurangi risiko overfitting yang umum terjadi pada pohon keputusan tunggal dan meningkatkan generalisasi model.

**2. Pembahasan Parameter**  
Model ini dioptimalkan menggunakan `GridSearchCV` untuk mencari kombinasi parameter terbaik. Parameter grid yang digunakan:
- `n_estimators`: [100, 200, 300] – Jumlah pohon dalam hutan
- `max_features`: ['sqrt', 'log2'] – Jumlah fitur yang dipertimbangkan dalam pemisahan node
- `max_depth`: [10, 20, None] – Kedalaman maksimum pohon
- `min_samples_split`: [2, 5] – Jumlah minimum sampel untuk membagi node internal
- `min_samples_leaf`: [1, 2] – Jumlah minimum sampel pada setiap daun pohon

**3. Kelebihan/Kekurangan**  
- **Kelebihan:** Sangat efektif dalam mengurangi overfitting, mampu menangani banyak fitur (termasuk fitur non-linier dan interaksi), dan dapat memberikan estimasi feature importance.  
- **Kekurangan:** Kurang dapat diinterpretasikan dibandingkan pohon keputusan tunggal karena kompleksitas banyak pohon. Proses pelatihan bisa lebih lambat jika jumlah pohon sangat banyak.

### 5.3 Model 3: XGBoost Classifier (Tuned)
**1. Pembahasan Cara Kerja**  
XGBoost (eXtreme Gradient Boosting) adalah algoritma boosting yang membangun pohon keputusan secara bertahap untuk memperbaiki kesalahan model sebelumnya. Setiap pohon fokus meminimalkan residual error. XGBoost mendukung regularisasi L1 dan L2 sehingga efektif mencegah overfitting, dan memiliki optimasi komputasi yang efisien.

**2. Pembahasan Parameter**  
Model ini dioptimalkan menggunakan `GridSearchCV` untuk mencari kombinasi hyperparameter terbaik. Parameter grid yang digunakan:
- `n_estimators`: [100, 200, 300]
- `learning_rate`: [0.01, 0.05, 0.1]
- `max_depth`: [3, 5, 7]
- `subsample`: [0.7, 0.8, 0.9]
- `colsample_bytree`: [0.7, 0.8, 0.9]
- `gamma`: [0, 0.1, 0.2]

Tuning dilakukan menggunakan `GridSearchCV` dengan `cv=5` (5-fold cross-validation) dan `scoring='accuracy'`. Parameter terbaik yang diperoleh (contoh hasil tuning):

```python
{
  'colsample_bytree': 0.9,
  'gamma': 0.1,
  'learning_rate': 0.05,
  'max_depth': 7,
  'n_estimators': 100,
  'subsample': 0.7
}
```
**3. Kelebihan/Kekurangan**
- **Kelebihan**: Performa sangat tinggi dan sering menjadi pemenang di berbagai kompetisi machine learning. Efisien dalam komputasi dan mampu menangani missing values secara internal. Memiliki fitur regularisasi yang kuat untuk mencegah overfitting.
- **Kekurangan**: Bisa kompleks untuk di-tune karena banyaknya hyperparameter. Model kurang mudah diinterpretasikan dibandingkan model linier atau pohon tunggal.


## 6. Evaluation

### 6.1. Evaluation Matrix yang Digunakan

Untuk menilai performa model klasifikasi biner, digunakan beberapa metrik utama berikut:

- **Accuracy**: Mengukur proporsi prediksi yang tepat dari keseluruhan data.
  
  $$
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  $$

- **Precision**: Menunjukkan seberapa akurat model dalam memprediksi kelas positif.
  
  $$
  \text{Precision} = \frac{TP}{TP + FP}
  $$

- **Recall**: Mengukur sejauh mana model berhasil mendeteksi semua data positif.
  
  $$
  \text{Recall} = \frac{TP}{TP + FN}
  $$

- **F1-Score**: Rata-rata harmonis dari precision dan recall, berguna ketika distribusi kelas tidak seimbang.
  
  $$
  F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  $$

- **Confusion Matrix**: Matriks yang memperlihatkan jumlah prediksi benar dan salah untuk tiap kelas:
  
  $$
  \begin{bmatrix}
  TN & FP \\
  FN & TP \\
  \end{bmatrix}
  $$

Pemilihan metrik ini disesuaikan dengan konteks bisnis, yaitu menghindari risiko anggur berkualitas rendah diklasifikasikan sebagai produk unggulan. Kesalahan ini dapat menurunkan kepuasan konsumen dan merusak reputasi merek.


### 6.2. Ringkasan Performa Model

Setelah proses pelatihan dan penyetelan parameter, hasil evaluasi pada data uji menunjukkan:

| Model                 | Akurasi  |
| :-------------------- | :------- |
| XGBoost (Tuned)       | 0.783088 |
| Random Forest (Tuned) | 0.772059 |
| Logistic Regression   | 0.735294 |

Model XGBoost menghasilkan akurasi tertinggi di antara ketiga model yang diuji.


### 6.3. Visualisasi Perbandingan Akurasi Model

Grafik berikut memperlihatkan perbandingan akurasi antar model secara visual:

![Perbandingan Akurasi Model](https://github.com/stnrliza/wine-quality-classification/blob/main/models-comparison.png?raw=true)

---

### 6.4. Analisis Model Terbaik: XGBoost (Tuned)

#### 6.4.1. Confusion Matrix dan Classification Report

Evaluasi XGBoost menunjukkan bahwa:

- Akurasi lebih tinggi dibanding model lainnya.
- Precision dan recall seimbang, menunjukkan model tidak berat sebelah.
- Confusion Matrix membantu mengidentifikasi kesalahan klasifikasi secara rinci.

Hal ini memperkuat keandalan XGBoost dalam mengklasifikasikan kualitas anggur secara konsisten dan tepat.

#### 6.4.2. Feature Importance

Visualisasi berikut menunjukkan kontribusi masing-masing fitur terhadap hasil prediksi XGBoost:

![Feature Importance](https://github.com/stnrliza/wine-quality-classification/blob/main/xgboost-feature-importances.png?raw=true)

Informasi ini memberikan wawasan penting bagi bisnis terkait fitur kimiawi yang paling memengaruhi kualitas anggur. Produsen dapat memanfaatkannya untuk meningkatkan pengawasan proses produksi dan menjaga mutu produk.


### 6.5. Business Understanding Correlation

Model yang dibangun telah berhasil menjawab problem statement dan tujuan utama, yaitu:

- Menghasilkan sistem klasifikasi otomatis untuk membedakan kualitas anggur secara efisien.
- Memberikan dasar pengambilan keputusan berbasis data dalam proses produksi.
- Mengurangi risiko distribusi anggur berkualitas rendah ke konsumen.

Dengan menggunakan model terbaik (XGBoost), prediksi kualitas dapat dilakukan secara lebih cepat dan akurat. Hal ini mendukung peningkatan kualitas layanan dan kepuasan pelanggan, serta memperkuat citra merek perusahaan di pasar.
