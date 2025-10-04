# =============================================================
#           DASHBOARD ANALISIS RS DENGAN AGENTIC AI
# =============================================================
# Deskripsi:
# Versi ini mengimplementasikan pipeline agentic (Plan-Draft-Critique-Finalize)
# dan menghasilkan laporan PDF yang lengkap, mencakup hasil akhir
# serta seluruh proses berpikir AI (rencana, draf, dan kritik).
# =============================================================

import streamlit as st
import pandas as pd
import json
import os
import io
from datetime import datetime

# --- Import Library Spesifik LLM ---
from groq import Groq, APIError, BadRequestError
from openai import OpenAI
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv

# --- Import Library untuk Membuat PDF ---
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import navy, black, gray

# --- Konfigurasi Awal ---
load_dotenv()
st.set_page_config(page_title="Dashboard Agentic AI", page_icon="🧠", layout="wide")


# ==========================================================
# BAGIAN 1: DEFINISI DAN KONFIGURASI PROVIDER AI
# ==========================================================
def _call_openai_compatible(client, model, temp, system, user):
    resp = client.chat.completions.create(model=model, temperature=temp, response_format={"type": "json_object"}, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return json.loads(resp.choices[0].message.content or "{}")

def _call_gemini(client, model, temp, system, user):
    prompt = f"{system}\n\n{user}\n\nOutput HANYA dalam format JSON yang valid."
    resp = client.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=temp, response_mime_type="application/json"))
    return json.loads(resp.text or "{}")

class ProviderQuotaError(Exception): pass

PROVIDER_CONFIG = {
    "OpenRouter": {"api_key_name": "OPENROUTER_API_KEY", "init_func": lambda key: OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1"), "call_func": _call_openai_compatible, "model": "meta-llama/llama-3.1-70b-instruct", "error_map": {(APIError, "insufficient_quota"): ProviderQuotaError}},
    "Groq": {"api_key_name": "GROQ_API_KEY", "init_func": lambda key: Groq(api_key=key), "call_func": _call_openai_compatible, "model": "llama-3.1-8b-instant", "error_map": {(APIError, "insufficient_quota"): ProviderQuotaError}},
    "Google": {"api_key_name": "GOOGLE_API_KEY", "init_func": lambda key: genai.configure(api_key=key) or genai.GenerativeModel("gemini-1.5-flash"), "call_func": _call_gemini, "model": "gemini-1.5-flash", "error_map": {(google_exceptions.ResourceExhausted, "free_tier"): ProviderQuotaError}}
}

ALL_POSSIBLE_PROVIDERS = ["OpenRouter", "Groq", "Google"]
default_available = []
for name in ALL_POSSIBLE_PROVIDERS:
    config = PROVIDER_CONFIG[name]
    api_key = os.getenv(config["api_key_name"]) or st.secrets.get(config["api_key_name"], "")
    is_available = bool(api_key)
    if is_available:
        try:
            config["client"] = config["init_func"](api_key)
        except Exception:
            is_available = False
    config["is_available"] = is_available
    if is_available:
        default_available.append(name)

# ==========================================================
# BAGIAN 2: UI SIDEBAR
# ==========================================================
st.sidebar.title("🕹️ Panel Kontrol")
st.sidebar.header("⚙️ Pengaturan LLM")

selected_providers = st.sidebar.multiselect("Pilih & Urutkan Prioritas Provider AI", options=ALL_POSSIBLE_PROVIDERS, default=default_available)
AVAILABLE_PROVIDERS = [p for p in selected_providers if PROVIDER_CONFIG[p].get("is_available")]

if AVAILABLE_PROVIDERS:
    st.sidebar.info(f"Urutan Fallback Aktif: {' → '.join(AVAILABLE_PROVIDERS)}")
else:
    st.sidebar.warning("Tidak ada provider AI yang aktif. Harap atur API Key Anda.")

st.session_state["openrouter_model"] = st.sidebar.selectbox("Model OpenRouter", ["meta-llama/llama-3.1-70b-instruct", "google/gemini-pro-1.5", "openai/gpt-4o"])
st.session_state["groq_model"] = st.sidebar.selectbox("Model Groq", ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"])
st.session_state["temperature"] = st.sidebar.slider("Temperature (Kreativitas)", 0.0, 1.0, 0.2, 0.1)

st.sidebar.header("🤖 Pengaturan Agentic")
max_critique_cycles = st.sidebar.slider("Jumlah Siklus Kritik", 0, 3, 1, 1)
st.sidebar.caption("Pastikan API keys diatur di file .env atau Streamlit Secrets.")


# ==========================================================
# BAGIAN 3: FUNGSI AGENTIC PIPELINE & PDF
# ==========================================================
# (Fungsi Agentic tidak berubah)
def proses_dengan_ai(system_prompt: str, user_prompt: str) -> dict:
    if not AVAILABLE_PROVIDERS: raise Exception("Tidak ada provider AI yang aktif.")
    # ... (logika fallback)
    last_error = None
    for provider_name in AVAILABLE_PROVIDERS:
        config = PROVIDER_CONFIG[provider_name]
        try:
            st.toast(f"Menggunakan provider: {provider_name}...")
            model_to_use = config["model"]
            if provider_name == "OpenRouter": model_to_use = st.session_state.get("openrouter_model", config["model"])
            elif provider_name == "Groq": model_to_use = st.session_state.get("groq_model", config["model"])
            result = config["call_func"](client=config["client"], model=model_to_use, temp=st.session_state.get("temperature", 0.3), system=system_prompt, user=user_prompt)
            result["_used_provider"] = f"{provider_name} ({model_to_use.split('/')[-1]})"
            return result
        except Exception as e:
            last_error = e
    raise Exception(f"Semua provider gagal. Error terakhir: {last_error}")

def agentic_plan(data_string: str, user_question: str) -> dict:
    system_prompt = "Anda adalah AI perencana strategis. Tugas Anda adalah membuat rencana analisis data yang terstruktur. Jawab HANYA dalam format JSON: {'rencana_analisis': ['langkah 1', 'langkah 2', '...']}"
    user_prompt = f"Berdasarkan data berikut dan permintaan pengguna, buatlah rencana analisis yang logis.\n\nPermintaan: {user_question}\n\nData (ringkasan): {data_string[:2000]}"
    return proses_dengan_ai(system_prompt, user_prompt)

def agentic_draft(data_string: str, user_question: str, plan: dict) -> dict:
    system_prompt = "Anda adalah analis data senior. Laksanakan rencana yang diberikan untuk menganalisis data dan menjawab permintaan pengguna. Jawab HANYA dalam format JSON: {'judul_analisis': '...', 'ringkasan_eksekutif': '...', 'draf_temuan': ['...'], 'draf_rekomendasi': ['...']}"
    user_prompt = f"Rencana: {plan}\n\nPermintaan: {user_question}\n\nData Lengkap: {data_string}"
    return proses_dengan_ai(system_prompt, user_prompt)

def agentic_critique(draft: dict) -> dict:
    system_prompt = "Anda adalah AI kritikus yang sangat teliti. Tinjau draf analisis ini, identifikasi kelemahan atau asumsi yang belum terbukti. Jawab HANYA dalam format JSON: {'kritik_dan_saran': ['...']}"
    user_prompt = f"Tolong berikan kritik konstruktif untuk draf analisis berikut:\n\n{json.dumps(draft, indent=2)}"
    return proses_dengan_ai(system_prompt, user_prompt)

def agentic_finalize(draft: dict, critique: dict) -> dict:
    system_prompt = "Anda adalah editor ahli. Perbaiki draf analisis awal berdasarkan kritik untuk menghasilkan laporan akhir yang lebih kuat. Jawab HANYA dalam format JSON yang sama persis dengan struktur draf awal."
    user_prompt = f"Draf Awal:\n{json.dumps(draft, indent=2)}\n\nKritik dan Saran:\n{json.dumps(critique, indent=2)}\n\nRevisi dan finalisasi draf tersebut."
    return proses_dengan_ai(system_prompt, user_prompt)

# [MODIFIKASI UTAMA] Fungsi Generate PDF yang lebih lengkap
def generate_pdf_report(full_result_data: dict) -> bytes:
    """Membuat laporan PDF dari seluruh proses agentic."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Styles
    styles = getSampleStyleSheet()
    style_h1 = styles['h1']
    style_h2 = ParagraphStyle(name='h2', parent=styles['h2'], fontSize=14, leading=16, spaceBefore=12, spaceAfter=6, textColor=navy)
    style_h3 = ParagraphStyle(name='h3', parent=styles['h3'], fontSize=11, leading=14, spaceBefore=10, spaceAfter=4, textColor=black)
    style_body = styles['BodyText']
    style_body.leading = 14
    style_code = ParagraphStyle(name='Code', parent=styles['Code'], fontSize=8, leading=10, backColor=gray, textColor=black, borderPadding=5, borderRadius=2)

    story = []

    # Helper untuk menambahkan konten ke PDF Story
    def add_to_story(text, style):
        story.append(Paragraph(text, style))
        story.append(Spacer(1, 0.1 * inch))

    def add_json_to_story(data, title):
        add_to_story(title, style_h3)
        # Ganti '<' dan '>' agar tidak dianggap sebagai tag HTML oleh ReportLab
        json_text = json.dumps(data, indent=2, ensure_ascii=False).replace('<', '&lt;').replace('>', '&gt;')
        story.append(Paragraph(f"<pre>{json_text}</pre>", style_code))
        story.append(Spacer(1, 0.2 * inch))

    # --- Menyusun Konten PDF ---
    final_result = full_result_data.get("main_result", {})
    
    # Halaman 1: Ringkasan
    add_to_story(final_result.get('judul_analisis', 'Laporan Analisis'), style_h1)
    
    add_to_story("Ringkasan Eksekutif", style_h2)
    add_to_story(final_result.get('ringkasan_eksekutif', 'Tidak ada ringkasan.'), style_body)

    add_to_story("Temuan Utama", style_h2)
    for finding in final_result.get("draf_temuan", ["-"]):
        add_to_story(f"• {finding}", style_body)

    add_to_story("Rekomendasi Aksi", style_h2)
    for rec in final_result.get("draf_rekomendasi", ["-"]):
        add_to_story(f"• {rec}", style_body)
        
    story.append(Spacer(1, 0.5 * inch))
    add_to_story(f"<i>Laporan ini dibuat pada {datetime.now().strftime('%d-%m-%Y %H:%M')} menggunakan {final_result.get('_used_provider', 'AI')}</i>", styles['Italic'])
    
    # Halaman Berikutnya: Proses Berpikir AI
    story.append(Paragraph("Lampiran: Proses Berpikir AI (Agentic Steps)", style_h1))
    story.append(Spacer(1, 0.2 * inch))

    add_json_to_story(full_result_data.get("plan", {}), "1. Rencana Analisis")
    add_json_to_story(full_result_data.get("initial_draft", {}), "2. Draf Awal")
    
    critiques = full_result_data.get("critiques", [])
    if critiques:
        for i, critique in enumerate(critiques):
            add_json_to_story(critique, f"3. Kritik & Saran (Siklus {i+1})")
            
    add_json_to_story(final_result, "4. Hasil Final (Setelah Revisi)")

    # Render PDF
    from reportlab.platypus import SimpleDocTemplate
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch)
    doc.build(story)
    
    buffer.seek(0)
    return buffer.getvalue()


# ==========================================================
# BAGIAN 4: TAMPILAN UTAMA APLIKASI
# ==========================================================
st.title("🧠 Dashboard Analisis dengan Agentic AI")
st.markdown("Unggah file Excel, ajukan pertanyaan, dan biarkan tim agen AI menganalisisnya untuk Anda.")

if 'data_sheets' not in st.session_state: st.session_state.data_sheets = None
if 'final_result' not in st.session_state: st.session_state.final_result = None

st.header("1. Unggah Bulk Excel")
uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx"])

if uploaded_file:
    try:
        st.session_state.data_sheets = pd.read_excel(uploaded_file, sheet_name=None)
        st.success(f"File **{uploaded_file.name}** berhasil diunggah.")
        st.session_state.final_result = None
    except Exception as e:
        st.error(f"Gagal membaca file Excel. Error: {e}")
        st.session_state.data_sheets = None

if st.session_state.data_sheets:
    st.markdown("---")
    st.header("🤖 2. Analisis Agentic AI")
    
    sheet_options = list(st.session_state.data_sheets.keys())
    selected_sheet = st.selectbox("Pilih sheet data:", sheet_options)
    user_question = st.text_area("Ajukan pertanyaan atau instruksi analisis:", height=100, placeholder="Contoh: Berikan analisis mendalam...")

    if st.button("🚀 Proses dengan Tim Agen AI", type="primary"):
        st.session_state.final_result = None
        if not user_question: st.error("Mohon ajukan pertanyaan.")
        elif not AVAILABLE_PROVIDERS: st.error("Tidak ada provider AI aktif.")
        else:
            try:
                # (Proses pipeline agentic tetap sama)
                df_to_analyze = st.session_state.data_sheets[selected_sheet]
                data_string = df_to_analyze.to_json(orient='split', indent=2)
                
                status = st.status("Tim Agen AI sedang bekerja...", expanded=True)
                status.write("LANGKAH 1: Merencanakan strategi analisis...")
                plan = agentic_plan(data_string, user_question)
                if not plan: raise Exception("Gagal membuat rencana.")
                
                status.write("LANGKAH 2: Menyusun draf awal...")
                draft = agentic_draft(data_string, user_question, plan)
                if not draft: raise Exception("Gagal membuat draf.")
                
                current_draft = draft
                critique_history = []
                for i in range(max_critique_cycles):
                    status.write(f"LANGKAH {3+i}: Meninjau draf (Siklus {i+1}/{max_critique_cycles})...")
                    critique = agentic_critique(current_draft)
                    if not critique or not critique.get('kritik_dan_saran'):
                        status.write("Kritik tidak menghasilkan saran, lanjut.")
                        break
                    critique_history.append(critique)
                    
                    status.write(f"LANGKAH {3+i}b: Merevisi draf...")
                    current_draft = agentic_finalize(current_draft, critique)
                    if not current_draft: raise Exception(f"Gagal memfinalisasi draf pada siklus {i+1}.")
                
                st.session_state.final_result = {"main_result": current_draft, "plan": plan, "initial_draft": draft, "critiques": critique_history}
                status.update(label="Analisis Selesai!", state="complete", expanded=False)

            except Exception as e:
                st.error(f"Terjadi kesalahan dalam pipeline agentic: {e}")
                st.session_state.final_result = None

if st.session_state.final_result:
    result_data = st.session_state.final_result
    final_result = result_data["main_result"]

    st.subheader(f"💡 Hasil Analisis Final: {final_result.get('judul_analisis', 'Tanpa Judul')}")
    st.success(f"Diselesaikan oleh: **{final_result.get('_used_provider', 'N/A')}**")

    st.markdown("**Ringkasan Eksekutif:**")
    st.write(f"> {final_result.get('ringkasan_eksekutif', 'Tidak ada.')}")

    st.markdown("**Temuan Utama:**")
    for temuan in final_result.get("draf_temuan", ["-"]):
        st.markdown(f"- {temuan}")

    st.markdown("**Rekomendasi Aksi:**")
    for rek in final_result.get("draf_rekomendasi", ["-"]):
        st.markdown(f"- {rek}")

    # Tombol Unduh PDF
    st.markdown("---")
    try:
        pdf_bytes = generate_pdf_report(result_data) # Kirim semua data proses ke fungsi PDF
        st.download_button(
            label="⬇️ Unduh Laporan Lengkap (PDF)",
            data=pdf_bytes,
            file_name=f"laporan_agentic_{selected_sheet.lower().replace(' ', '_')}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Gagal membuat file PDF: {e}")

    # Expander
    with st.expander("🔍 Lihat Proses Berpikir AI (Agentic Steps)"):
        st.markdown("#### 1. Rencana Analisis"); st.json(result_data["plan"])
        st.markdown("#### 2. Draf Awal"); st.json(result_data["initial_draft"])
        if result_data["critiques"]:
            for i, critique in enumerate(result_data["critiques"]):
                st.markdown(f"#### 3. Kritik & Saran (Siklus {i+1})"); st.json(critique)
        st.markdown("#### 4. Hasil Final (Setelah Revisi)"); st.json(final_result)
else:
    if not uploaded_file:
        st.info("Silakan unggah file Excel untuk memulai analisis.")

