import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog
from tkinter import ttk
import numpy as np
import os

# Import tambahan untuk PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Konstanta Matematika Sesuai Ketentuan Dokumen Soal
E_CONST = 2.71828183

def sigmoid(x):
    return 1 / (1 + (E_CONST ** (-x)))

def sigmoid_derivative(output):
    return output * (1 - output)

class ModernBackpropGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Proyek Akhir (Tugas UAS) AI ANN")
        
        # Dimensi jendela diatur agar seimbang dan proporsional
        self.root.geometry("1400x920")
        self.root.minimum_size = (1300, 850)
        self.root.configure(bg="#0B0F19") 

        # --- KONFIGURASI STYLE TTK UNTUK DARK MODE ---
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TLabelframe", background="#111827", relief="solid", borderwidth=1)
        self.style.configure("TLabelframe.Label", font=("Segoe UI", 11, "bold"), foreground="#F3F4F6", background="#111827")

        # --- HEADER UTAMA ---
        header_frame = tk.Frame(root, bg="#111827", pady=15, relief="solid", borderwidth=1)
        header_frame.pack(fill="x", side="top")
        
        tk.Label(header_frame, text="🏢 PREDIKSI KEBANGKRUTAN", font=("Segoe UI", 22, "bold"), fg="#F9FAFB", bg="#111827").pack()
        tk.Label(header_frame, text="🧠 Jaringan Syaraf Tiruan Backpropagation | Oleh: ADI DARMANA (310125023966)", 
                 font=("Segoe UI", 10, "normal"), fg="#38BDF8", bg="#111827").pack(pady=5)

        # Kontainer Konten Utama (Menggunakan Grid Sistem agar Pembagian Kolom Proporsional Akurat)
        content_container = tk.Frame(root, bg="#0B0F19", padx=20, pady=15)
        content_container.pack(fill="both", expand=True)

        content_container.columnconfigure(0, weight=45, uniform="main_layout") # Kolom Kiri (Input & Bobot)
        content_container.columnconfigure(1, weight=55, uniform="main_layout") # Kolom Kanan (Tombol & Log)
        content_container.rowconfigure(0, weight=1)

        # ==================== [ SEBELAH KIRI: AREA INPUT & BOBOT AWAL ] ====================
        kiri_frame = tk.Frame(content_container, bg="#0B0F19")
        kiri_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # 1. Jendela Parameter Utama Model ANN
        param_lf = ttk.LabelFrame(kiri_frame, text="  ⚙️ PARAMETER UTAMA MODEL ANN  ")
        param_lf.pack(fill="x", pady=(0, 10))
        
        p_pad = tk.Frame(param_lf, bg="#111827", padx=15, pady=10)
        p_pad.pack(fill="both", expand=True)

        lbl_cfg = {"font": ("Segoe UI", 10, "bold"), "bg": "#111827", "fg": "#9CA3AF"}
        ent_cfg = {"font": ("Segoe UI", 10), "relief": "solid", "borderwidth": 1, "bg": "#1F2937", "fg": "#F9FAFB", "justify": "center", "insertbackground": "white"}

        tk.Label(p_pad, text="🪪 NIM Pengali :", **lbl_cfg).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_nim = tk.Entry(p_pad, width=20, **ent_cfg)
        self.entry_nim.insert(0, "310125023966")
        self.entry_nim.grid(row=0, column=1, padx=5, pady=5, ipady=2, sticky="w")
        self.entry_nim.bind("<KeyRelease>", self.on_nim_changed)

        tk.Label(p_pad, text="🔄 Max Epoch :", **lbl_cfg).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry_epoch = tk.Entry(p_pad, width=20, **ent_cfg)
        self.entry_epoch.insert(0, "1054")  
        self.entry_epoch.grid(row=1, column=1, padx=5, pady=5, ipady=2, sticky="w")

        tk.Label(p_pad, text="🎯 Target Error :", **lbl_cfg).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entry_error = tk.Entry(p_pad, width=20, **ent_cfg)
        self.entry_error.insert(0, "0.01")
        self.entry_error.grid(row=2, column=1, padx=5, pady=5, ipady=2, sticky="w")

        tk.Label(p_pad, text="📐 Konstanta e :", **lbl_cfg).grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.entry_e_const = tk.Entry(p_pad, width=20, **ent_cfg)
        self.entry_e_const.insert(0, "2.71828183")
        self.entry_e_const.grid(row=3, column=1, padx=5, pady=5, ipady=2, sticky="w")
        
        # [PERBAIKAN] Tambahan input untuk Learning Rate (α)
        tk.Label(p_pad, text="📈 Learning Rate (α) :", **lbl_cfg).grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.entry_lr = tk.Entry(p_pad, width=20, **ent_cfg)
        self.entry_lr.insert(0, "1") 
        self.entry_lr.grid(row=4, column=1, padx=5, pady=5, ipady=2, sticky="w")

        self.lbl_aturan = tk.Label(p_pad, text="", font=("Segoe UI", 9, "italic"), bg="#111827", fg="#FB7185", justify="left", wraplength=480)
        self.lbl_aturan.grid(row=5, column=0, columnspan=2, padx=5, pady=(8, 0), sticky="w")

        # 2. Jendela Dataset Training Manual
        data_lf = ttk.LabelFrame(kiri_frame, text="  📅 DATASET TRAINING MANUAL  ")
        data_lf.pack(fill="x", pady=(0, 10))
        
        d_pad = tk.Frame(data_lf, bg="#111827", padx=15, pady=10)
        d_pad.pack(fill="both", expand=True)
        d_pad.columnconfigure(1, weight=1)
        d_pad.columnconfigure(2, weight=1)
        d_pad.columnconfigure(3, weight=1)

        th_cfg = {"font": ("Segoe UI", 9, "bold"), "bg": "#111827", "fg": "#6B7280"}
        tk.Label(d_pad, text="📥 Pendapatan (x1)", **th_cfg).grid(row=0, column=1, padx=4, pady=2)
        tk.Label(d_pad, text="📥 Hutang (x2)", **th_cfg).grid(row=0, column=2, padx=4, pady=2)
        tk.Label(d_pad, text="📍 Target (t)", **th_cfg).grid(row=0, column=3, padx=4, pady=2)

        tk.Label(d_pad, text="Data 1 :", **lbl_cfg).grid(row=1, column=0, padx=4, pady=4, sticky="w")
        self.x11 = tk.Entry(d_pad, width=10, **ent_cfg); self.x11.insert(0, "0.9"); self.x11.grid(row=1, column=1, padx=2, pady=4, ipady=2, sticky="ew")
        self.x12 = tk.Entry(d_pad, width=10, **ent_cfg); self.x12.insert(0, "0.4"); self.x12.grid(row=1, column=2, padx=2, pady=4, ipady=2, sticky="ew")
        self.t1  = tk.Entry(d_pad, width=10, **ent_cfg); self.t1.insert(0, "1.0");  self.t1.grid(row=1, column=3, padx=2, pady=4, ipady=2, sticky="ew")

        tk.Label(d_pad, text="Data 2 :", **lbl_cfg).grid(row=2, column=0, padx=4, pady=4, sticky="w")
        self.x21 = tk.Entry(d_pad, width=10, **ent_cfg); self.x21.insert(0, "0.73"); self.x21.grid(row=2, column=1, padx=2, pady=4, ipady=2, sticky="ew")
        self.x22 = tk.Entry(d_pad, width=10, **ent_cfg); self.x22.insert(0, "0.85"); self.x22.grid(row=2, column=2, padx=2, pady=4, ipady=2, sticky="ew")
        self.t2  = tk.Entry(d_pad, width=10, **ent_cfg); self.t2.insert(0, "0.0");  self.t2.grid(row=2, column=3, padx=2, pady=4, ipady=2, sticky="ew")

        # 3. Keterangan Bobot Awal Acak
        bobot_lf = ttk.LabelFrame(kiri_frame, text="  📋 KETERANGAN BOBOT AWAL (ACAK) & HASIL SKALASI NIM  ")
        bobot_lf.pack(fill="both", expand=True)

        self.txt_bobot_view = scrolledtext.ScrolledText(bobot_lf, wrap=tk.WORD, font=("Consolas", 9), 
                                                        bg="#1F2937", fg="#E5E7EB", relief="flat")
        self.txt_bobot_view.pack(fill="both", expand=True, padx=10, pady=10)


        # ==================== [ SEBELAH KANAN: TOMBOL & MONITORING CONSOLE LOG ] ====================
        kanan_frame = tk.Frame(content_container, bg="#0B0F19")
        kanan_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        # Tombol aksi utama dipasang di paling atas sebelah kanan
        self.btn_hitung = tk.Button(kanan_frame, text="⚡ JALANKAN PROSES KOMPUTASI BACKPROPAGATION", command=self.proses_backpropagation, 
                                    bg="#059669", fg="#FFFFFF", activebackground="#047857", activeforeground="#FFFFFF",
                                    font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2", bd=0, pady=12)
        self.btn_hitung.pack(fill="x", side="top", pady=(0, 10))
        
        self.btn_hitung.bind("<Enter>", lambda e: self.btn_hitung.config(bg="#047857"))
        self.btn_hitung.bind("<Leave>", lambda e: self.btn_hitung.config(bg="#059669"))

        # Jendela Besar Pembungkus Monitor Log Perhitungan
        output_lf = ttk.LabelFrame(kanan_frame, text="  🖥️ MONITORING CONSOLE LOG PERHITUNGAN  ")
        output_lf.pack(fill="both", expand=True, side="top")

        # Menggunakan struktur Grid di dalam area log untuk memisahkan box teks dengan input keyword pencarian secara mutlak
        output_lf.columnconfigure(0, weight=1)
        output_lf.rowconfigure(0, weight=1)  # Area Log mengambil sisa ruang vertikal terbanyak
        output_lf.rowconfigure(1, weight=0)  # Area Pencarian & Cetak mengambil ruang minimal

        # Kotak cetak output Log Utama
        self.txt_log = scrolledtext.ScrolledText(output_lf, wrap=tk.WORD, font=("Consolas", 10), 
                                                 bg="#030712", fg="#10B981", insertbackground="#FFFFFF",
                                                 padx=15, pady=15, relief="flat")
        self.txt_log.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Toolbar Pencarian Teks Log & Tombol Cetak PDF
        tools_frame = tk.Frame(output_lf, bg="#111827", pady=6)
        tools_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        tk.Label(tools_frame, text="🔍 Cari Keyword :", font=("Segoe UI", 10, "bold"), bg="#111827", fg="#9CA3AF").pack(side="left", padx=(10, 5))
        self.entry_search = tk.Entry(tools_frame, font=("Segoe UI", 10), width=25, relief="solid", borderwidth=1, bg="#1F2937", fg="#F9FAFB", insertbackground="white")
        self.entry_search.pack(side="left", ipady=3, padx=(0, 15))
        self.entry_search.bind("<KeyRelease>", self.cari_teks_log)

        # --- TOMBOL CETAK PDF BARU ---
        self.btn_pdf = tk.Button(tools_frame, text="🖨️ CETAK LOG KE PDF", command=self.cetak_log_ke_pdf,
                                 bg="#3B82F6", fg="#FFFFFF", activebackground="#2563EB", activeforeground="#FFFFFF",
                                 font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2", padx=15, pady=3)
        self.btn_pdf.pack(side="right", padx=(0, 10))
        self.btn_pdf.bind("<Enter>", lambda e: self.btn_pdf.config(bg="#2563EB"))
        self.btn_pdf.bind("<Leave>", lambda e: self.btn_pdf.config(bg="#3B82F6"))

        # Inisialisasi penampilan awal teks aturan & data bobot di box kiri
        self.hitung_dan_tampilkan_bobot_statis()

    def on_nim_changed(self, event=None):
        self.hitung_dan_tampilkan_bobot_statis()

    def hitung_dan_tampilkan_bobot_statis(self):
        nim_str = self.entry_nim.get().strip()
        if len(nim_str) >= 4:
            four_digits = nim_str[-4:]
            multiplier = float(f"0.{four_digits}")
            text_aturan = f"⚠️ ATURAN: 4 digit terakhir NIM ({four_digits}) = 0.{four_digits} → pengali (multiplier) untuk input & bobot awal. Target (t) tetap."
            self.lbl_aturan.config(text=text_aturan)
        else:
            self.lbl_aturan.config(text="⚠️ ATURAN: Masukkan minimal 4 digit NIM untuk menghitung pengali.")
            multiplier = 1.0

        # Definisi Bobot Mentah Sesuai Soal
        v_raw = np.array([[0.9562, 0.7762, 0.1623, 0.2886], [0.1962, 0.6133, 0.0311, 0.9711]])
        v0_raw = np.array([0.7496, 0.3796, 0.7256, 0.1628])
        w_raw = np.array([[0.2280], [0.9585], [0.6799], [0.0550]])
        w0_raw = np.array([[0.9505]])

        # Hitung Bobot Terkalibrasi
        v = v_raw * multiplier
        v0 = v0_raw * multiplier
        w = w_raw * multiplier
        w0 = w0_raw * multiplier

        # Render ke Widget Box Teks di sebelah kiri secara Real-Time
        self.txt_bobot_view.configure(state='normal')
        self.txt_bobot_view.delete(1.0, tk.END)
        
        info_text = f"""• Bobot input ke hidden (Acak Mentah):
  v11=0.9562   v12=0.7762   v13=0.1623   v14=0.2886 
  v21=0.1962   v22=0.6133   v23=0.0311   v24=0.9711
  ↳ [Hasil setelah dikalikan {multiplier:.4f}]:
  v11={v[0][0]:.6f}  v12={v[0][1]:.6f}  v13={v[0][2]:.6f}  v14={v[0][3]:.6f}
  v21={v[1][0]:.6f}  v22={v[1][1]:.6f}  v23={v[1][2]:.6f}  v24={v[1][3]:.6f}

------------------------------------------------------------------
• Bobot bias ke hidden (Acak Mentah):
  V01=0.7496   V02=0.3796   V03=0.7256   V04=0.1628
  ↳ [Hasil setelah dikalikan {multiplier:.4f}]:
  V01={v0[0]:.6f}  V02={v0[1]:.6f}  V03={v0[2]:.6f}  V04={v0[3]:.6f}

------------------------------------------------------------------
• Bobot hidden ke output (Acak Mentah):
  w1=0.2280    w2=0.9585    w3=0.6799    w4=0.0550
  ↳ [Hasil setelah dikalikan {multiplier:.4f}]:
  w1 ={w[0][0]:.6f}  w2 ={w[1][0]:.6f}  w3 ={w[2][0]:.6f}  w4 ={w[3][0]:.6f}

------------------------------------------------------------------
• Bobot bias ke output (Acak Mentah):
  w0=0.9505   
  ↳ [Hasil setelah dikalikan {multiplier:.4f}]: w0={w0[0][0]:.6f}"""
        
        self.txt_bobot_view.insert(tk.END, info_text)
        self.txt_bobot_view.configure(state='disabled')

    def cari_teks_log(self, event=None):
        self.txt_log.tag_remove('match', '1.0', tk.END)
        query = self.entry_search.get().strip()
        
        if query:
            start_pos = '1.0'
            while True:
                start_pos = self.txt_log.search(query, start_pos, stopindex=tk.END, nocase=True)
                if not start_pos:
                    break
                end_pos = f"{start_pos}+{len(query)}c"
                self.txt_log.tag_add('match', start_pos, end_pos)
                self.txt_log.tag_config('match', background="#1E3A8A", foreground="#F3F4F6")
                start_pos = end_pos

    def cetak_log_ke_pdf(self):
        """Fungsi baru untuk mengekspor teks log perhitungan ke dalam file PDF terformat."""
        log_content = self.txt_log.get(1.0, tk.END).strip()
        
        if not log_content or log_content == "MULAI PERHITUNGAN BACKPROPAGATION":
            messagebox.showwarning("Log Kosong", "Belum ada log perhitungan. Silakan jalankan komputasi terlebih dahulu!")
            return

        # Meminta lokasi penyimpanan file dari user
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")],
            title="Simpan Log Perhitungan Sebagai PDF",
            initialfile=f"Log_Backprop_{self.entry_nim.get().strip()}.pdf"
        )
        
        if not file_path:
            return  # Jika user membatalkan dialog penyimpanan

        try:
            # Setup Dokumen
            doc = SimpleDocTemplate(file_path, pagesize=letter,
                                    rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
            story = []
            styles = getSampleStyleSheet()

            # Custom Style untuk Log Monospace agar layout rapi (simbol kotak/tabel terminal dikonversi)
            log_style = ParagraphStyle(
                'LogStyle',
                parent=styles['Normal'],
                fontName='Courier',
                fontSize=8.5,
                leading=11,
                textColor=colors.HexColor('#111827')
            )
            
            title_style = ParagraphStyle(
                'TitleStyle',
                fontName='Helvetica-Bold',
                fontSize=16,
                leading=20,
                textColor=colors.HexColor('#059669'),
                alignment=1 # Center
            )

            # Judul Dokumen PDF
            story.append(Paragraph("LAPORAN LOG KOMPUTASI BACKPROPAGATION", title_style))
            story.append(Spacer(1, 15))

            # Proses teks log baris per baris agar ReportLab membacanya dengan benar sebagai paragraf XML
            lines = log_content.split('\n')
            for line in lines:
                # Sanitasi string agar tidak merusak parser XML ReportLab
                processed_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                # Mengganti spasi beruntun menjadi non-breaking space (&nbsp;) agar indentasi tabulasi terjaga
                processed_line = processed_line.replace(' ', '&nbsp;')
                
                if processed_line.strip() == "":
                    story.append(Spacer(1, 4))
                else:
                    story.append(Paragraph(processed_line, log_style))

            # Build file PDF
            doc.build(story)
            messagebox.showinfo("Sukses", f"Log Perhitungan berhasil dicetak ke PDF!\n\nLokasi: {file_path}")

        except Exception as e:
            messagebox.showerror("Gagal Cetak PDF", f"Terjadi kesalahan saat membuat PDF:\n\nDetail: {str(e)}")

    def proses_backpropagation(self):
        nim_str = self.entry_nim.get().strip()
        
        try:
            max_epoch = int(self.entry_epoch.get())
            target_error = float(self.entry_error.get())
            # [PERBAIKAN] Mengambil learning_rate dari UI
            learning_rate = float(self.entry_lr.get()) 
            
            # Membaca nilai dinamis konstanta e dari GUI Entry
            e_const_val = float(self.entry_e_const.get())
            
            # Membuat fungsi sigmoid lokal berdasarkan nilai e yang dimasukkan
            def dynamic_sigmoid(x):
                return 1 / (1 + (e_const_val ** (-x)))
            
            if len(nim_str) < 4:
                raise ValueError("NIM harus memiliki minimal 4 digit terakhir.")
            
            in_x11 = float(self.x11.get())
            in_x12 = float(self.x12.get())
            in_t1  = float(self.t1.get())
            
            in_x21 = float(self.x21.get())
            in_x22 = float(self.x22.get())
            in_t2  = float(self.t2.get())
            
            four_digit_nim = nim_str[-4:]
            multiplier = float(f"0.{four_digit_nim}")
            
        except ValueError as e:
            messagebox.showerror("Kesalahan Input", f"Format angka tidak valid.\n\nDetail Error: {e}")
            return

        self.txt_log.delete(1.0, tk.END)
        # [PERBAIKAN] Baris ini DIHAPUS karena akan mengacaukan fungsi learning_rate sebenarnya
        # learning_rate = target_error 
        
        # Ambil bobot awal
        v_raw = np.array([[0.9562, 0.7762, 0.1623, 0.2886], [0.1962, 0.6133, 0.0311, 0.9711]])
        v0_raw = np.array([0.7496, 0.3796, 0.7256, 0.1628])
        w_raw = np.array([[0.2280], [0.9585], [0.6799], [0.0550]])
        w0_raw = np.array([[0.9505]])

        v = v_raw * multiplier
        v0 = v0_raw * multiplier
        w = w_raw.copy() * multiplier
        w0 = w0_raw.copy() * multiplier

        # 1. HEADER REKAP DATA (Ke Monitor Log di Sebelah Kanan)
        self.txt_log.insert(tk.END, " MULAI PERHITUNGAN BACKPROPAGATION\n")
        self.txt_log.insert(tk.END, "==============================================================================\n")
        self.txt_log.insert(tk.END, f"NIM           : {nim_str}  |  4 digit: {four_digit_nim}\n")
        self.txt_log.insert(tk.END, f"Multiplier    : {multiplier:.4f}\n")
        self.txt_log.insert(tk.END, f"Learning Rate : {learning_rate}\n")
        self.txt_log.insert(tk.END, f"Max Epoch     : {max_epoch}\n")
        self.txt_log.insert(tk.END, f"Konstanta e   : {e_const_val:.8f}\n")
        self.txt_log.insert(tk.END, "------------------------------------------------------------------------------\n\n")

        # 2. DATA TRAINING (SETELAH SKALASI MULTIPLIER)
        x11_scaled, x12_scaled = in_x11 * multiplier, in_x12 * multiplier
        x21_scaled, x22_scaled = in_x21 * multiplier, in_x22 * multiplier
        
        self.txt_log.insert(tk.END, "DATA TRAINING (setelah skalasi):\n")
        self.txt_log.insert(tk.END, f"  Data 1: x1={x11_scaled:.6f}, x2={x12_scaled:.6f}, t={int(in_t1)}\n")
        self.txt_log.insert(tk.END, f"  Data 2: x1={x21_scaled:.6f}, x2={x22_scaled:.6f}, t={int(in_t2)}\n\n")

        X = np.array([[x11_scaled, x12_scaled], [x21_scaled, x22_scaled]])
        target = np.array([[in_t1], [in_t2]])
        
        status_sukses = False
        epoch_terhenti = max_epoch
        total_mse = 0.0

        # --- LOOPS TRAINING EPOCH ---
        for epoch in range(1, max_epoch + 1):
            epoch_loss = 0
            
            self.txt_log.insert(tk.END, "-------------|-----------------------------------------\n")
            self.txt_log.insert(tk.END, f"             |  EPOCH {epoch} dari {max_epoch}\n")
            self.txt_log.insert(tk.END, "-------------|-----------------------------------------\n\n")

            for i in range(len(X)):
                self.txt_log.insert(tk.END, f"  === Data {i+1} ===\n\n")
                
                # --- FORWARD PASS ---
                self.txt_log.insert(tk.END, "  --- FORWARD PASS ---\n")
                self.txt_log.insert(tk.END, "  [1] Hidden Layer:\n")
                
                z_in = []
                z = []
                for j in range(4):
                    z_in_val = v0[j] + (X[i][0] * v[0][j]) + (X[i][1] * v[1][j])
                    z_val = dynamic_sigmoid(z_in_val)
                    z_in.append(z_in_val)
                    z.append(z_val)
                    
                    self.txt_log.insert(tk.END, f"    z_in{j+1} = {v0[j]:.6f} + ({X[i][0]:.6f}x{v[0][j]:.6f}) + ({X[i][1]:.6f}x{v[1][j]:.6f})\n")
                    self.txt_log.insert(tk.END, f"            = {v0[j]:.6f} + {X[i][0]*v[0][j]:.6f} + {X[i][1]*v[1][j]:.6f} = {z_in_val:.6f}\n")
                    self.txt_log.insert(tk.END, f"    z_{j+1}  = sigmoid({z_in_val:.6f}) = {z_val:.6f}\n")

                y_in_val = w0[0][0] + (z[0] * w[0][0]) + (z[1] * w[1][0]) + (z[2] * w[2][0]) + (z[3] * w[3][0])
                y_val = dynamic_sigmoid(y_in_val)
                
                error_val = target[i][0] - y_val
                loss_val = error_val ** 2
                epoch_loss += loss_val
                
                self.txt_log.insert(tk.END, "\n  [2] Output Layer:\n")
                self.txt_log.insert(tk.END, f"    y_in = {w0[0][0]:.6f} + {z[0]:.6f}x{w[0][0]:.6f} + {z[1]:.6f}x{w[1][0]:.6f} + {z[2]:.6f}x{w[2][0]:.6f} + {z[3]:.6f}x{w[3][0]:.6f}\n")
                self.txt_log.insert(tk.END, f"         = {y_in_val:.6f}\n")
                self.txt_log.insert(tk.END, f"    y    = sigmoid({y_in_val:.6f}) = {y_val:.6f}\n")
                
                self.txt_log.insert(tk.END, "\n  [3] Error:\n")
                self.txt_log.insert(tk.END, f"    Error = {int(target[i][0])} - {y_val:.6f} = {error_val:.6f}\n")
                self.txt_log.insert(tk.END, f"    MSE   = ({error_val:.6f})^2 = {loss_val:.6f}\n\n")

                # --- BACKWARD PASS ---
                self.txt_log.insert(tk.END, "  --- BACKWARD PASS ---\n")
                self.txt_log.insert(tk.END, "  [1] Delta Output Layer:\n")
                y_deriv = y_val * (1 - y_val)
                d_out = error_val * y_deriv
                self.txt_log.insert(tk.END, f"    y x (1-y) = {y_val:.6f} x {1-y_val:.6f} = {y_deriv:.6f}\n")
                self.txt_log.insert(tk.END, f"    d_out = error x y x (1-y) = {error_val:.6f} x {y_deriv:.6f} = {d_out:.6f}\n\n")

                w_old_epoch_data = w.copy()

                self.txt_log.insert(tk.END, "  [2] Update Bobot Hidden -> Output:\n")
                dw0 = learning_rate * d_out
                w0_old = w0[0][0]
                w0[0][0] += dw0
                self.txt_log.insert(tk.END, f"    Dw0 = {learning_rate} x {d_out:.6f} = {dw0:.6f}\n")
                self.txt_log.insert(tk.END, f"    w0: {w0_old:.6f} -> {w0[0][0]:.6f}\n")
                
                for j in range(4):
                    dw = learning_rate * d_out * z[j]
                    w_old = w[j][0]
                    w[j][0] += dw
                    self.txt_log.insert(tk.END, f"    Dw{j+1} = {learning_rate} x {d_out:.6f} x {z[j]:.6f} = {dw:.6f}\n")
                    self.txt_log.insert(tk.END, f"    w{j+1}: {w_old:.6f} -> {w[j][0]:.6f}\n")

                self.txt_log.insert(tk.END, "\n  [3] Propagasi Error ke Hidden (d_hidden):\n")
                dh = []
                for j in range(4):
                    d_in_j = d_out * w_old_epoch_data[j][0] 
                    z_deriv = z[j] * (1 - z[j])
                    d_j = d_in_j * z_deriv
                    dh.append(d_j)
                    self.txt_log.insert(tk.END, f"    d_in{j+1} = {d_out:.6f} x {w_old_epoch_data[j][0]:.6f} = {d_in_j:.6f}\n")
                    self.txt_log.insert(tk.END, f"    d_{j+1}  = {d_in_j:.6f} x {z[j]:.6f} x {1-z[j]:.6f} = {d_j:.6f}\n")

                # [PERBAIKAN] Log Detail untuk Update Bobot Input -> Hidden
                self.txt_log.insert(tk.END, "\n  [4] Update Bobot Input -> Hidden:\n")
                for j in range(4):
                    dv1 = learning_rate * dh[j] * X[i][0]
                    dv2 = learning_rate * dh[j] * X[i][1]
                    dv0 = learning_rate * dh[j]
                    
                    v1_old, v2_old, v0_old = v[0][j], v[1][j], v0[j]
                    
                    v[0][j] += dv1
                    v[1][j] += dv2
                    v0[j] += dv0
                    
                    self.txt_log.insert(tk.END, f"    z{j+1}:\n")
                    self.txt_log.insert(tk.END, f"         Dv1{j+1} = {learning_rate} x {dh[j]:.6f} x {X[i][0]:.6f} = {dv1:.6f}\n")
                    self.txt_log.insert(tk.END, f"         Dv2{j+1} = {learning_rate} x {dh[j]:.6f} x {X[i][1]:.6f} = {dv2:.6f}\n")
                    self.txt_log.insert(tk.END, f"         DV0{j+1} = {learning_rate} x {dh[j]:.6f} = {dv0:.6f}\n")
                    
                    self.txt_log.insert(tk.END, f"         v1{j+1}  : {v1_old:.6f} -> {v[0][j]:.6f}\n")
                    self.txt_log.insert(tk.END, f"         v2{j+1}  : {v2_old:.6f} -> {v[1][j]:.6f}\n")
                    self.txt_log.insert(tk.END, f"         V0{j+1}  : {v0_old:.6f} -> {v0[j]:.6f}\n")
                self.txt_log.insert(tk.END, "\n")

            total_mse = epoch_loss / len(X)
            self.txt_log.insert(tk.END, "  ----------------------------------------\n")
            self.txt_log.insert(tk.END, f"  HASIL EPOCH {epoch}: Total MSE = {total_mse:.6f}\n")
            self.txt_log.insert(tk.END, "  ----------------------------------------\n\n\n")
            
            if total_mse <= target_error:
                status_sukses = True
                epoch_terhenti = epoch
                break

        self.txt_log.see(tk.END)
        
        if status_sukses:
            messagebox.showinfo(
                "📢 Perhitungan Selesai", 
                f"Proses Backpropagation Selesai Sempurna!\n\n"
                f"Target Error ({target_error}) telah terpenuhi lebih awal pada Epoch ke-{epoch_terhenti}.\n"
                f"Nilai Rata-Rata MSE Terakhir: {total_mse:.6f}.\n\n"
                f"Klik [OK] untuk menutup pesan ini."
            )
        else:
            messagebox.showinfo(
                "📢 Perhitungan Selesai", 
                f"Proses Iterasi Selesai!\n\n"
                f"Seluruh {max_epoch} Epoch yang Anda minta telah tuntas dihitung.\n"
                f"Nilai Rata-Rata MSE Terakhir: {total_mse:.6f}.\n\n"
                f"Klik [OK] untuk menutup pesan ini."
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernBackpropGUI(root)
    root.mainloop()