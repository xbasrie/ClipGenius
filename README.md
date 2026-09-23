# ClipGenius 🎬⚡

**Local-First AI Video Repurposing Studio** — Otomatisasi pemotongan video panjang (YouTube / file lokal) menjadi video pendek viral vertikal (TikTok, IG Reels, YouTube Shorts) dengan subtitle dinamis dan pelacakan pembicara otomatis.

Didesain untuk berjalan langsung di komputer lokal Windows Anda tanpa ketergantungan langganan SaaS cloud berbayar.

---

## 🚀 Fitur Utama

- **Smart Video Downloader & Metadata Extractor:** Mengunduh video YouTube resolusi tinggi dan membaca metadata secara otomatis via engine `yt-dlp` terintegrasi.
- **Bypass Transkripsi Cepat:** Otomatis mengambil subtitle resmi/komunitas YouTube (`json3`) jika tersedia, menghemat waktu transkripsi dari puluhan menit menjadi hitungan detik.
- **Local STT (Speech-to-Text) Whisper:** Transkripsi offline dengan `faster-whisper` (CTranslate2) jika video tidak memiliki subtitle. Mendukung akselerasi GPU (CUDA) maupun mode efisien CPU INT8.
- **Smart Viral Curation Engine:**
  - **Mode LLM (Gemini / OpenAI / Ollama / 9Router):** Membaca konteks transkrip penuh untuk menemukan momen dengan hook pembuka kuat, jeda alami (natural cut), dan payoff cerita tuntas.
  - **Mode Heuristic (100% Offline Tanpa API Key):** Algoritma penilaian matematis berdasarkan kepadatan kata (*word density*), kata pemicu atensi (*hook markers*), dan data kuantitatif (*numbers*).
- **Auto-Reframe Cerdas (16:9 ➔ 9:16 Vertikal):**
  - Pelacakan wajah pembicara menggunakan OpenCV Haar Cascade dengan dynamic smoothing (*SmoothedCameraman*) agar panning kamera tidak patah-patah.
  - Mode *Blurred General Background* otomatis saat tidak ada pembicara tunggal.
- **Studio Editor & Live Customization:**
  - **Timeline Multi-Track & Scrubber:** Navigasi seekbar halus bergaya CapCut.
  - **Pilihan Font & Tipografi Visual:** Dilengkapi preview langsung bentuk font (Montserrat, Impact, Bebas Neue, Anton, Poppins, Inter, Komika).
  - **Teks Hook Atas (Header Video):** Input dan pratinjau live judul/sticker hook penarik atensi di bagian atas video.
  - **Penata Subtitle Lengkap:** Slider ukuran font, posisi vertikal (Y), warna teks, dan background label opsional.
  - **Editor Transkrip Kata:** Edit dan koreksi teks kata langsung sebelum diekspor.
  - **Filmstrip Klip Proyek:** Kartu navigasi antar klip dengan thumbnail, badge skor viral, dan durasi.
- **Ekspor Video Instan:** Render video MP4 vertikal 1080p berkualitas tinggi dengan audio jernih dan subtitle permanen (*hardsubbed*).

---

## 💡 Peningkatan dari Repo Sebelumnya (Improvement over OpenShorts)

| Fitur / Aspek | OpenShorts Asli | ClipGenius |
| :--- | :--- | :--- |
| **Arsitektur & Runtime** | CLI / Skrip Python murni | **Desktop GUI Native + REST Sidecar FastAPI** |
| **Aksesibilitas** | Terminal / command line | **Studio Editor Webview Interaktif** |
| **Ketergantungan API** | Wajib API Key Gemini | **Hybrid:** Mendukung LLM cloud/lokal **+ Heuristic Fallback 100% Offline** |
| **Efisiensi Transkripsi** | Selalu menjalankan Whisper dari awal | **Auto-Bypass:** Pakai subtitle bawaan YouTube jika ada (hemat 90% waktu) |
| **Pratinjau Tipografi** | Blind select (hanya teks nama font) | **Visual Font Cards:** Menampilkan sampel asli tipografi dan bobot font |
| **Hook Banner Header** | Statis tanpa preview | **Live Interactive Hook Overlay** di atas canvas player |
| **Stabilitas Face Tracking**| Rawan crash jika file Haar Cascade hilang | **Fail-Safe Cascade:** Fallback mulus tanpa membatalkan proses render |
| **Distribusi** | Setup manual dependensi | **Tersedia binary mandiri `.exe`** (PyInstaller siap pakai) |

---

## 🖥️ Kebutuhan Perangkat (System Requirements)

### Spesifikasi Minimum (Mode CPU):
- **Sistem Operasi:** Windows 10 / 11 (64-bit)
- **Prosesor (CPU):** Intel Core i3 (Generasi 8+) atau AMD Ryzen 3 / quad-core 2.5 GHz+
- **Memori (RAM):** 8 GB
- **Penyimpanan:** Minimal 10 GB ruang kosong (direkomendasikan SSD)
- **GPU:** Integrated Intel UHD Graphics / AMD Radeon Graphics (pemrosesan via CPU)

### Spesifikasi Rekomendasi (Akselerasi GPU / Fast Render):
- **Sistem Operasi:** Windows 11 (64-bit)
- **Prosesor (CPU):** Intel Core i5/i7 (Generasi 11+) atau AMD Ryzen 5/7 (6 core atau lebih)
- **Memori (RAM):** 16 GB atau lebih
- **Kartu Grafis (GPU):** NVIDIA GeForce GTX 1650 / RTX 3050 ke atas (VRAM minimal 4 GB dengan dukungan CUDA)
- **Penyimpanan:** NVMe M.2 SSD untuk pemrosesan video berkecepatan tinggi

### Dependensi Eksternal:
- **FFmpeg & FFprobe:** Terpasang di sistem (`PATH`) atau di folder `resources/bin/`.
- **Node.js:** Terpasang di sistem (diperlukan `yt-dlp` untuk parsing JavaScript YouTube).

---

## 🛠️ Instalasi & Menjalankan Aplikasi

### Opsi A: Menjalankan Binary Desktop (.exe)
1. Unduh atau buka build rilis di folder:
   ```
   dist\ClipGenius\ClipGenius.exe
   ```
   *(atau klik shortcut `ClipGenius.lnk` di Desktop).*
2. Aplikasi otomatis memeriksa ketersediaan port, menjalankan background engine, dan membuka window antarmuka Studio di browser atau jendela Edge App.

### Opsi B: Menjalankan dari Source Code (Python)
1. **Clone repositori:**
   ```bash
   git clone https://github.com/xbasrie/ClipGenius.git
   cd ClipGenius
   ```

2. **Buat & aktifkan virtual environment:**
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # Windows (Git Bash):
   source .venv/Scripts/activate
   ```

3. **Install dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi desktop:**
   ```bash
   python desktop_main.py
   ```
   Buka `http://127.0.0.1:8089/` di browser pilihan Anda.

---

## ⚙️ Konfigurasi LLM (Opsional)

Jika ingin menggunakan kurasi berbasis LLM selain kurasi lokal:
Buka tab **Pengaturan** di aplikasi dan masukkan:
- **Provider:** `gemini`, `openai`, `ollama`, atau `custom` (misal 9Router / Local Router).
- **API Key & Base URL:** Masukkan kredensial provider yang Anda gunakan.
- Jika dikosongkan, sistem secara otomatis beralih ke engine **Heuristic Offline** tanpa biaya token.

---

## 📂 Struktur Proyek

```
ClipGenius/
├── desktop_main.py          # Entrypoint desktop GUI & port management
├── pipeline/
│   ├── server.py            # FastAPI sidecar backend & background worker
│   ├── config.py            # Konfigurasi path, FFmpeg, dan yt-dlp
│   ├── db.py                # Database SQLite (jobs, clips, settings)
│   ├── modulos/
│   │   ├── media.py         # Downloader yt-dlp & probe durasi/resolusi
│   │   ├── stt.py           # Engine Whisper Speech-to-Text
│   │   ├── curation.py      # Kurasi momen viral (LLM + Heuristic)
│   │   ├── clip.py          # Auto-reframe 9:16 & OpenCV speaker tracking
│   │   ├── subtitle.py      # Generator subtitle ASS / SRT
│   │   └── export.py        # Rendering final video FFMPEG & hook filter
│   └── static/
│       └── index.html       # Antarmuka studio modern (Tailwind + Lucide)
├── resources/
│   ├── bin/                 # Executable pembantu (yt-dlp wrapper)
│   └── haarcascades/        # Model klasifikasi wajah OpenCV
└── tests/                   # Smoke tests & server API tests
```

---

## 📄 Lisensi

Didistribusikan di bawah lisensi MIT. Silakan gunakan, kembangkan, dan sesuaikan dengan kebutuhan alur kerja konten Anda.
