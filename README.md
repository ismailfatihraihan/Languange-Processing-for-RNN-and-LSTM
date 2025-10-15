# Proyek NLP: Analisis Sentimen dan Chatbot dengan RNN & LSTM

Notebook ini merupakan sebuah proyek Natural Language Processing (NLP) yang mengimplementasikan model Recurrent Neural Network (RNN) dan Long Short-Term Memory (LSTM) untuk dua tugas utama dalam Bahasa Indonesia:
1.  **Analisis Sentimen**: Mengklasifikasikan teks ke dalam sentimen positif atau negatif.
2.  **Chatbot Q&A**: Membangun model sekuens-ke-sekuens (Seq2Seq) untuk menjawab pertanyaan medis dan psikiatri.

## Alur Kerja Notebook

Notebook ini terstruktur secara sistematis, dimulai dari persiapan data hingga evaluasi model.

### 1. Pemuatan dan Persiapan Data

-   **Mount Google Drive**: Mengakses file dataset yang tersimpan di Google Drive.
-   **Instalasi Pustaka**: Menginstal `Sastrawi` untuk proses stemming Bahasa Indonesia.
-   **Pemuatan Dataset**:
    -   **Data Sentimen**: Memuat 3 dataset CSV (`dataset_komentar_instagram_cyberbullying.csv`, `dataset_tweet_sentimen_tayangan_tv.csv`, `dataset_tweet_sentiment_opini_film.csv`). Kolom-kolom yang relevan diseragamkan menjadi `text` dan `sentiment`.
    -   **Data Q&A**: Memuat 2 dataset (`qna-medical.csv`, `qna-psychiatrist.parquet`). Kolom-kolomnya diseragamkan menjadi `question` dan `answer`.
    -   **Pengambilan Sampel**: Untuk data Q&A, dibuat sebuah dataset sampel strategis berukuran 50.000 baris yang terdiri dari seluruh data psikiater dan sampel acak dari data medis.
    -   **File Tambahan**: Memuat `stop_words`, `slang_words`, dan `wordlist` dari file `.txt` untuk digunakan dalam tahap pra-pemrosesan.

### 2. Pra-pemrosesan Teks (Preprocessing)

Sebuah fungsi `preprocess_text_cached` dibuat untuk membersihkan dan menstandarisasi semua teks. Proses ini mencakup:
-   **Case Folding**: Mengubah semua teks menjadi huruf kecil.
-   **Cleaning**: Menghapus URL, mention (@), hashtag (#), angka, dan karakter non-alfabet.
-   **Normalisasi**: Mengganti kata-kata slang dengan bentuk bakunya (contoh: `ga` -> `tidak`).
-   **Stopword Removal**: Menghapus kata-kata umum yang tidak memiliki makna signifikan (contoh: `adalah`, `di`, `dan`).
-   **Stemming**: Mengubah kata ke bentuk dasarnya menggunakan pustaka `Sastrawi` (contoh: `memakan` -> `makan`).

Hasil dari pra-pemrosesan disimpan dalam file format `.feather` (`qna_50k_clean.feather`, `sentiment_clean.feather`) untuk mempercepat pemuatan di sesi berikutnya.

### 3. Tokenisasi dan Padding

Teks yang sudah bersih diubah menjadi format numerik yang dapat diproses oleh model deep learning.
-   **Tokenizer**: Setiap kata unik dalam korpus data diberi sebuah indeks integer. Kata-kata yang tidak ada dalam vocabulary akan ditandai dengan token `<OOV>` (Out-of-Vocabulary).
-   **Sequencing**: Setiap kalimat diubah menjadi urutan (sekuens) dari indeks-indeks integer.
-   **Padding**: Semua sekuens disamakan panjangnya dengan menambahkan nilai nol di akhir (`post-padding`).

### 4. Pelatihan Model: Analisis Sentimen

Eksperimen dilakukan dengan membandingkan dua arsitektur utama.
-   **SimpleRNN**: Dilatih 3 variasi model untuk membandingkan efek regularisasi dan optimizer. Model dengan regularisasi dan optimizer `RMSprop` menunjukkan performa terbaik dengan akurasi validasi sekitar 75%.
-   **LSTM**: Dilatih 4 variasi model (baseline, kapasitas tinggi, learning rate rendah, dan stacked). Model-model LSTM secara konsisten gagal belajar dari data ini, dengan akurasi validasi stagnan di sekitar 50%.
-   **Kesimpulan**: Untuk dataset sentimen ini, model **SimpleRNN** yang lebih sederhana terbukti lebih efektif daripada LSTM.

### 5. Pelatihan Model: Chatbot Q&A (Seq2Seq)

Model chatbot dibangun menggunakan arsitektur Encoder-Decoder.
-   **Eksperimen Arsitektur**:
    1.  **LSTM Baseline**: Model dasar dengan 128 unit LSTM.
    2.  **SimpleRNN Baseline**: Model dasar dengan 128 unit SimpleRNN.
    3.  **LSTM High Capacity**: Model dengan kapasitas lebih besar (256 unit LSTM). Model ini menunjukkan performa terbaik dari keempatnya.
    4.  **Stacked LSTM**: Model dengan dua lapis LSTM.
-   **Eksperimen Data & Embedding**:
    1.  **Pre-trained Embeddings**: Model terbaik (LSTM High Capacity) dilatih kembali menggunakan word embedding FastText yang sudah dilatih sebelumnya pada korpus besar Bahasa Indonesia.
    2.  **Model Spesialis**: Sebuah model baru dilatih hanya menggunakan dataset psikiatri untuk melihat peningkatan performa pada domain yang lebih sempit.
    3.  **Pembersihan Pola Sapaan**: Dilakukan pembersihan tambahan pada jawaban data medis untuk menghilangkan sapaan berulang (misal: "Halo, terima kasih telah bertanya di Alodokter").
    4.  **Bidirectional LSTM**: Arsitektur encoder diperkuat dengan menggunakan Bidirectional LSTM untuk menangkap konteks dari dua arah.

### 6. Evaluasi

-   **Kuantitatif**: Model dievaluasi menggunakan metrik `loss` dan `accuracy` pada data test.
-   **Kualitatif**: Kemampuan model chatbot diuji dengan memberikan beberapa pertanyaan sampel dan membandingkan jawaban yang dihasilkan oleh model-model yang berbeda. Terdapat juga sel interaktif untuk melakukan "live chat" dengan model yang telah dilatih.

### 7. Referensi Dataset
**Opini Pilkada DKI**  
Lestari, A.R.T., Perdana, R.S., & Fauzi, M.A. (2017). Analisis Sentimen Tentang Opini Pilkada DKI 2017 Pada Dokumen Twitter Berbahasa Indonesia Menggunakan Naïve Bayes dan Pembobotan Emoji. Jurnal Pengembangan Teknologi Informasi Dan Ilmu Komputer, 1(12), 1718-1724. Diambil dari http://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/627  
**Sentimen Twitter**  
Rofiqoh, U., Perdana, R.S., & Fauzi, M.A. (2017). Analisis Sentimen Tingkat Kepuasan Pengguna Penyedia Layanan Telekomunikasi Seluler Indonesia Pada Twitter Dengan Metode Support Vector Machine dan Lexicon Based Features. Jurnal Pengembangan Teknologi Informasi Dan Ilmu Komputer, 1(12), 1725-1732. Diambil dari http://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/628  
**Cyber Bullying**  
Luqyana, W., Cholissodin, I., & Perdana, R.S. (2018). Analisis Sentimen Cyberbullying pada Komentar Instagram dengan Metode Klasifikasi Support Vector Machine. Jurnal Pengembangan Teknologi Informasi Dan Ilmu Komputer, 2(11), 4704-4713. Diambil dari http://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/3051  
**Sentimen TV**  
Nurjanah, W.E., Perdana, R.S., & Fauzi, M.A. (2017). Analisis Sentimen Terhadap Tayangan Televisi Berdasarkan Opini Masyarakat pada Media Sosial Twitter menggunakan Metode K-Nearest Neighbor dan Pembobotan Jumlah Retweet. Jurnal Pengembangan Teknologi Informasi Dan Ilmu Komputer, 1(12), 1750-1757. Diambil dari http://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/631  
**Opini Film**  
Antinasari, P., Perdana, R.S., & Fauzi, M.A. (2017). Analisis Sentimen Tentang Opini Film Pada Dokumen Twitter Berbahasa Indonesia Menggunakan Naive Bayes Dengan Perbaikan Kata Tidak Baku. Jurnal Pengembangan Teknologi Informasi Dan Ilmu Komputer, 1(12), 1733-1741. Diambil dari http://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/629  


