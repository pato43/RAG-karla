# Streamlit RAG — estructura base estable
# -------------------------------------------------------------
# Enfoque: núcleo RAG limpio y robusto (sin calendario/bitácora)
# - Carga PDFs/Imágenes
# - Extracción de texto (PDF con PyMuPDF; OCR opcional con EasyOCR)
# - Metadatos heurísticos (fecha, folio/UUID, RFC, total)
# - Indexado TF‑IDF + recuperación Top‑K (Escala K)
# - QA con LLM vía OpenRouter
# Requisitos (requirements.txt):
#   streamlit
#   pandas
#   numpy
#   requests
#   python-dateutil
#   scikit-learn
#   pillow
#   pymupdf
#   easyocr     # opcional (si no está, el OCR se omite)
# -------------------------------------------------------------

import os
import io
import json
import base64
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import requests
from dateutil import parser as dateparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# PDF
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# OCR (opcional, sin LLM)
try:
    import easyocr
except Exception:
    easyocr = None

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN BASE + TEMA
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG de Documentos — Base", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
      :root { --fg:#0f172a; --muted:#64748b; --primary:#2563eb; }
      .app-header {background: linear-gradient(135deg,#eef2ff,#e0f2fe); padding:18px 22px; border-radius:16px; border:1px solid #e5e7eb}
      .app-title {font-size:1.6rem; font-weight:800; color:var(--fg)}
      .app-sub {color:var(--muted)}
      .metric-card {border:1px solid #e5e7eb; border-radius:14px; padding:12px; background:#fff}
      .metric-card h3{font-size:.85rem; color:#475569; margin:0 0 6px}
      .metric-card .val{font-size:1.3rem; font-weight:800}
      .stTabs [data-baseweb="tab"]{padding:10px 16px; border-radius:10px; background:#f8fafc; border:1px solid #e2e8f0}
      .stTabs [aria-selected="true"]{background:#e0f2fe !important; border-color:#bae6fd !important}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# ESTADO DE SESIÓN
# ──────────────────────────────────────────────────────────────────────────────
if "docs" not in st.session_state:
    # Cada doc: {id, nombre, origen, texto, paginas, metadata:{fecha, folio, rfc, total, etiquetas}, ts}
    st.session_state.docs: List[Dict[str, Any]] = []
if "index" not in st.session_state:
    st.session_state.index = None  # {vectorizer, matrix, mapping}
if "settings" not in st.session_state:
    st.session_state.settings = {
        "project_name": "RAG-Base",
        "model": "openai/gpt-4o-mini",   # modelo de OpenRouter
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 5,                       # Escala K
        "origen": "Cargados manualmente",
        "force_ocr": False,
    }

# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE LOG / ID
# ──────────────────────────────────────────────────────────────────────────────

def make_id() -> str:
    return base64.urlsafe_b64encode(os.urandom(9)).decode("utf-8").rstrip("=")


# ──────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE METADATOS
# ──────────────────────────────────────────────────────────────────────────────
DATE_REGEXES = [r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"]
UUID_REGEX = r"\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\b"
RFC_REGEX = r"\b([A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3})\b"  # RFC MX aproximado
TOTAL_REGEX = r"(?i)\b(total|importe|monto)\b\D{0,15}([\$\s]*[0-9]+[\,\.]?[0-9]{0,2})"
FOLIO_HINTS = r"(?i)(folio|factura|no\.?\s*\w*|id|referencia)\s*[:#\-]?\s*([A-Za-z0-9\-]{5,})"


def parse_dates(text: str) -> str | None:
    for rgx in DATE_REGEXES:
        m = re.search(rgx, text)
        if m:
            try:
                d = dateparser.parse(m.group(1), dayfirst=True, fuzzy=True)
                return d.date().isoformat()
            except Exception:
                pass
    return None


def extract_metadata(text: str, filename: str) -> Dict[str, Any]:
    md = {"nombre": filename, "fecha": parse_dates(text) or None, "folio": None, "rfc": None, "total": None, "etiquetas": ""}
    m = re.search(UUID_REGEX, text)
    if m:
        md["folio"] = m.group(0)
    else:
        m2 = re.search(FOLIO_HINTS, text)
        if m2:
            md["folio"] = m2.group(2)[:40]
    m = re.search(RFC_REGEX, text)
    if m:
        md["rfc"] = m.group(1)
    m = re.search(TOTAL_REGEX, text)
    if m:
        md["total"] = m.group(2).strip()
    return md

# ──────────────────────────────────────────────────────────────────────────────
# OCR Y EXTRACCIÓN DE TEXTO
# ──────────────────────────────────────────────────────────────────────────────

_OCR_READER = None

def get_easyocr_reader():
    global _OCR_READER
    if _OCR_READER is None and easyocr is not None:
        try:
            _OCR_READER = easyocr.Reader(["es", "en"], gpu=False)
        except Exception:
            _OCR_READER = None
    return _OCR_READER


def ocr_image(img: Image.Image) -> str:
    reader = get_easyocr_reader()
    if reader is None:
        return "(OCR no disponible: easyocr no instalado)"
    arr = np.array(img)
    result = reader.readtext(arr, detail=0, paragraph=True)
    return "\n".join([t.strip() for t in result if t and t.strip()])


def pdf_bytes_to_text_and_preview(b: bytes, force_ocr: bool = False) -> Tuple[str, Image.Image | None, int]:
    if not fitz:
        return "", None, 0
    textos: List[str] = []
    preview_img = None
    try:
        with fitz.open(stream=b, filetype="pdf") as doc:
            for i, page in enumerate(doc):
                page_text = page.get_text("text")
                needs_ocr = force_ocr or (not page_text or len(page_text.strip()) < 10)
                if needs_ocr and easyocr is not None:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_text = ocr_image(img)
                    if i == 0:
                        preview_img = img
                else:
                    if i == 0:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        preview_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                textos.append(page_text)
        return "\n".join(textos), preview_img, len(textos)
    except Exception as e:
        return f"[ERROR PDF] {e}", None, 0


def image_bytes_to_text_and_preview(b: bytes) -> Tuple[str, Image.Image | None]:
    try:
        img = Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return "", None
    text = ocr_image(img) if easyocr is not None else "(OCR desactivado)"
    return text, img

# ──────────────────────────────────────────────────────────────────────────────
# RAG: CHUNKING, INDEXADO, RETRIEVAL
# ──────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        slice_text = text[start:end]
        last_punct = max(slice_text.rfind(". "), slice_text.rfind("\n"))
        if last_punct > 100:  # evita cortes demasiado cortos
            end = start + last_punct + 1
        chunks.append(text[start:end])
        if end >= L:
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c.strip()]


def build_index(documents: List[Dict[str, Any]], chunk_size: int, overlap: int):
    corpus = []
    mapping = []  # (doc_id, chunk_idx)
    for d in documents:
        for i, ch in enumerate(chunk_text(d.get("texto", ""), chunk_size, overlap)):
            corpus.append(ch)
            mapping.append({"doc_id": d["id"], "chunk_idx": i})
    if not corpus:
        return None
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
    matrix = vectorizer.fit_transform(corpus)
    st.session_state.index = {"vectorizer": vectorizer, "matrix": matrix, "mapping": mapping}
    return st.session_state.index


def retrieve(query: str, top_k: int, origin_filter: List[str] | None = None) -> List[Dict[str, Any]]:
    idx = st.session_state.index
    if not idx:
        return []
    qv = idx["vectorizer"].transform([query])
    sims = cosine_similarity(qv, idx["matrix"]).ravel()
    order = np.argsort(-sims)
    results = []
    for ti in order:
        m = idx["mapping"][ti]
        doc = next((d for d in st.session_state.docs if d["id"] == m["doc_id"]), None)
        if doc and (not origin_filter or doc.get("origen") in origin_filter):
            chunks = chunk_text(doc.get("texto", ""), st.session_state.settings["chunk_size"], st.session_state.settings["chunk_overlap"]) or [""]
            ctx = chunks[m["chunk_idx"]] if m["chunk_idx"] < len(chunks) else ""
            results.append({"score": float(sims[ti]), "doc": doc, "chunk": ctx})
            if len(results) >= top_k:
                break
    return results

# ──────────────────────────────────────────────────────────────────────────────
# OPENROUTER (LLM)
# ──────────────────────────────────────────────────────────────────────────────

def get_openrouter_key() -> str | None:
    return st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")


def openrouter_chat(messages: List[Dict[str, Any]], model: str | None = None, temperature: float = 0.1, max_tokens: int = 800) -> str:
    key = get_openrouter_key()
    if not key:
        st.warning("⚠️ Agrega OPENROUTER_API_KEY en Secrets o variable de entorno.")
        return ""
    model = model or st.session_state.settings["model"]
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "RAG-Base",
    }
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    except Exception as e:
        st.error(f"Error LLM: {e}")
        return ""

# ──────────────────────────────────────────────────────────────────────────────
# UI — CABECERA
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="app-header"><div class="app-title">🧠 RAG de Documentos — Base</div><div class="app-sub">Carga, extrae, indexa y pregunta a tus documentos. OCR sin LLM opcional.</div></div>', unsafe_allow_html=True)

# Sidebar de configuración
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.session_state.settings["project_name"] = st.text_input("Proyecto", st.session_state.settings["project_name"])
    st.session_state.settings["model"] = st.text_input("Modelo (OpenRouter)", st.session_state.settings["model"], help="Ej: openai/gpt-4o-mini, anthropic/claude-3.5-sonnet, meta/llama-3.1-70b-instruct")
    st.session_state.settings["chunk_size"] = st.slider("Tamaño de chunk", 400, 2000, st.session_state.settings["chunk_size"], step=50)
    st.session_state.settings["chunk_overlap"] = st.slider("Solapamiento", 0, 400, st.session_state.settings["chunk_overlap"], step=20)
    st.session_state.settings["top_k"] = st.slider("Escala K (Top‑K)", 1, 10, st.session_state.settings["top_k"])
    st.session_state.settings["origen"] = st.selectbox("Origen actual", ["Cargados manualmente", "Google Drive (próx)", "Correo (próx)", "S3 (próx)"])
    st.session_state.settings["force_ocr"] = st.toggle("Forzar OCR en PDFs sin texto", value=st.session_state.settings["force_ocr"])
    st.divider()
    st.markdown("**Secrets necesarios**")
    st.code("OPENROUTER_API_KEY='sk-or-...""", language="bash")

# ──────────────────────────────────────────────────────────────────────────────
# TABS PRINCIPALES
# ──────────────────────────────────────────────────────────────────────────────

TAB_DOCS, TAB_RAG = st.tabs(["📥 Documentos", "🔎 RAG & QA"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB: DOCUMENTOS
# ──────────────────────────────────────────────────────────────────────────────
with TAB_DOCS:
    st.subheader("Carga y extracción")
    ups = st.file_uploader("Arrastra PDFs/Imágenes (png, jpg, webp)", type=["pdf", "png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

    if ups:
        for f in ups:
            b = f.read()
            doc_id = make_id()
            texto, preview, paginas = "", None, 0
            if f.type == "application/pdf":
                texto, preview, paginas = pdf_bytes_to_text_and_preview(b, force_ocr=st.session_state.settings["force_ocr"])
            else:
                texto, preview = image_bytes_to_text_and_preview(b)
                paginas = 1

            md = extract_metadata(texto or "", f.name)
            doc = {
                "id": doc_id,
                "nombre": md.get("nombre") or f.name,
                "origen": st.session_state.settings.get("origen"),
                "texto": texto or "",
                "paginas": paginas,
                "metadata": {
                    "fecha": md.get("fecha"),
                    "folio": md.get("folio"),
                    "rfc": md.get("rfc"),
                    "total": md.get("total"),
                    "etiquetas": "",
                },
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            st.session_state.docs.append(doc)

            with st.expander(f"📄 {f.name}"):
                cprev, cmeta = st.columns([1, 2])
                with cprev:
                    if preview is not None:
                        st.image(preview, caption=f"Vista previa ({paginas} pág)", use_container_width=True)
                with cmeta:
                    st.write("**Metadatos (editable)**")
                    editable = st.data_editor(
                        pd.DataFrame([{ "id": doc_id, "Nombre": doc["nombre"], "Fecha": doc["metadata"]["fecha"], "Folio": doc["metadata"]["folio"], "RFC": doc["metadata"]["rfc"], "Total": doc["metadata"]["total"], "Etiquetas": doc["metadata"]["etiquetas"], "Origen": doc["origen"] }]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    row = editable.iloc[0]
                    doc["nombre"] = row["Nombre"]
                    doc["origen"] = row["Origen"]
                    doc["metadata"].update({"fecha": row["Fecha"], "folio": row["Folio"], "rfc": row["RFC"], "total": row["Total"], "etiquetas": row["Etiquetas"]})
                    st.write("**Texto extraído (vista)**")
                    st.text_area("", (texto or "")[:3500], height=220)

    st.divider()
    st.subheader("Tabla de documentos")
    if st.session_state.docs:
        df = pd.DataFrame([{ "ID": d["id"], "Nombre": d["nombre"], "Origen": d["origen"], "Fecha": d["metadata"].get("fecha"), "Folio": d["metadata"].get("folio"), "RFC": d["metadata"].get("rfc"), "Total": d["metadata"].get("total"), "Páginas": d.get("paginas", 0), "Cargado": d.get("ts"), "Etiquetas": d["metadata"].get("etiquetas", "") } for d in st.session_state.docs])
        edited = st.data_editor(df, hide_index=True, use_container_width=True)
        for _, row in edited.iterrows():
            d = next((x for x in st.session_state.docs if x["id"] == row["ID"]), None)
            if d:
                d["nombre"] = row["Nombre"]
                d["origen"] = row["Origen"]
                d["metadata"].update({"fecha": row["Fecha"], "folio": row["Folio"], "rfc": row["RFC"], "total": row["Total"], "etiquetas": row.get("Etiquetas", "")})
        st.download_button("⬇️ Exportar CSV", edited.to_csv(index=False).encode("utf-8"), file_name="documentos.csv", mime="text/csv")
    else:
        st.info("Carga documentos para comenzar.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB: RAG & QA
# ──────────────────────────────────────────────────────────────────────────────
with TAB_RAG:
    st.subheader("Indexado y preguntas al LLM")
    left, right = st.columns([1,2])

    with left:
        if st.button("🧱 (Re)construir índice", type="primary"):
            idx = build_index(st.session_state.docs, st.session_state.settings["chunk_size"], st.session_state.settings["chunk_overlap"])
            if idx:
                st.success(f"Índice con {idx['matrix'].shape[0]} chunks.")
            else:
                st.warning("No hay texto indexable.")
        st.caption("Usa la Escala K en Config para ajustar el número de pasajes.")

    with right:
        colQ1, colQ2 = st.columns([2,1])
        with colQ1:
            q = st.text_input("Pregunta", placeholder="¿Cuál es el total de la factura X?")
        with colQ2:
            origenes = sorted(set(d.get("origen") for d in st.session_state.docs))
            filtro_origen = st.multiselect("Origen", origenes)

        if q:
            ctxs = retrieve(q, st.session_state.settings["top_k"], origin_filter=filtro_origen if filtro_origen else None) if st.session_state.index else []
            if not ctxs:
                st.warning("Construye el índice o carga documentos.")
            else:
                context_blob = "\n\n".join([f"[Doc: {c['doc']['nombre']} | Fecha: {c['doc']['metadata'].get('fecha')} | Folio: {c['doc']['metadata'].get('folio')}]\n{c['chunk']}" for c in ctxs])
                messages = [
                    {"role": "system", "content": "Responde usando SOLO el contexto. Cuando cites, incluye Nombre/Fecha/Folio si aparece. Si falta evidencia, indica que no está en los documentos."},
                    {"role": "user", "content": f"Contexto:\n{context_blob}\n\nPregunta: {q}"},
                ]
                ans = openrouter_chat(messages, model=st.session_state.settings["model"], temperature=0.1, max_tokens=700)
                st.markdown("### Respuesta")
                st.write(ans or "(sin respuesta)")
                pack = {
                    "question": q,
                    "answer": ans,
                    "used": [
                        {"doc": c["doc"]["nombre"], "fecha": c["doc"]["metadata"].get("fecha"), "folio": c["doc"]["metadata"].get("folio"), "score": c["score"]}
                        for c in ctxs
                    ],
                }
                st.download_button("💾 Guardar respuesta (JSON)", json.dumps(pack, ensure_ascii=False, indent=2), "respuesta.json")
                with st.expander("Contexto usado"):
                    for i, c in enumerate(ctxs, 1):
                        meta = c['doc']['metadata']
                        st.markdown(f"**{i}. {c['doc']['nombre']}** · score={c['score']:.3f} · Fecha: {meta.get('fecha')} · Folio: {meta.get('folio')}")
                        st.write(c["chunk"][:1600])

# Fin — versión base, lista para integrar calendario/login/bitácora si lo necesitas.
