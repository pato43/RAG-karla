import os
import io
import json
import base64
import re
from datetime import datetime, date, timedelta
from typing import List, Dict, Any

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

# Calendario (componente de terceros basado en FullCalendar)
try:
    from streamlit_calendar import calendar
except Exception:
    calendar = None

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN BASE
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG de Documentos + Agenda",
    page_icon="📁",
    layout="wide",
)

# Estilos ligeros para una UI limpia y profesional
st.markdown(
    """
    <style>
      .app-title {font-size: 1.8rem; font-weight: 700; margin-bottom: .25rem}
      .app-sub {color: #666; margin-bottom: 1rem}
      .metric-badge {display:inline-block; padding: 4px 10px; border-radius: 8px; background:#F1F5F9; margin-right:6px}
      .pill {padding:2px 8px; border-radius:999px; background:#EEF2FF; font-size:12px; margin-left:6px}
      .stTabs [data-baseweb="tab-list"] {gap: 8px}
      .stTabs [data-baseweb="tab"] {padding: 10px 16px; border-radius: 8px; background: #F8FAFC}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# SESIÓN Y ESTADO
# ──────────────────────────────────────────────────────────────────────────────
if "auth" not in st.session_state:
    st.session_state.auth = False
if "user" not in st.session_state:
    st.session_state.user = None
if "docs" not in st.session_state:
    # Cada doc: {id, nombre, origen, texto, paginas, metadata:{fecha, folio, rfc, total}, ts}
    st.session_state.docs: List[Dict[str, Any]] = []
if "index" not in st.session_state:
    st.session_state.index = None  # {vectorizer, matrix, mapping}
if "events" not in st.session_state:
    st.session_state.events = []  # calendario
if "log" not in st.session_state:
    st.session_state.log: List[Dict[str, Any]] = []  # bitácora
if "settings" not in st.session_state:
    st.session_state.settings = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 5,
        "model": "openai/gpt-4o-mini",  # cualquiera disponible en OpenRouter
        "vision_model": "openai/gpt-4o-mini",  # opcional para imágenes
        "project_name": "Demo-RAG",
    }

# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────────────────────

def log_event(tipo: str, detalle: str, extra: Dict[str, Any] | None = None):
    st.session_state.log.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tipo": tipo,
        "detalle": detalle,
        "extra": extra or {},
        "usuario": st.session_state.user,
    })


def make_id() -> str:
    return base64.urlsafe_b64encode(os.urandom(9)).decode("utf-8").rstrip("=")


DATE_REGEXES = [
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
]
UUID_REGEX = r"\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\b"
RFC_REGEX = r"\b([A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3})\b"  # RFC mexicano aproximado
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
                continue
    return None


def extract_metadata(text: str, filename: str) -> Dict[str, Any]:
    md = {
        "nombre": filename,
        "fecha": parse_dates(text) or None,
        "folio": None,
        "rfc": None,
        "total": None,
    }
    # Folio / UUID / hints
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
        # Grupo 2 tiene la cifra probable
        md["total"] = m.group(2).strip()

    return md


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        # procura cortar en punto o salto cercano
        slice_text = text[start:end]
        last_punct = max(slice_text.rfind(". "), slice_text.rfind("\n"))
        if last_punct > 100:  # evita cortes muy cortos
            end = start + last_punct + 1
        chunks.append(text[start:end])
        start = max(end - overlap, 0)
        if start == end:
            break
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


def retrieve(query: str, top_k: int) -> List[Dict[str, Any]]:
    idx = st.session_state.index
    if not idx:
        return []
    qv = idx["vectorizer"].transform([query])
    sims = cosine_similarity(qv, idx["matrix"]).ravel()
    top_idx = np.argsort(-sims)[:top_k]
    results = []
    for ti in top_idx:
        m = idx["mapping"][ti]
        doc = next((d for d in st.session_state.docs if d["id"] == m["doc_id"]), None)
        if doc:
            chunk = chunk_text(doc.get("texto", ""), st.session_state.settings["chunk_size"], st.session_state.settings["chunk_overlap"]) or [""]
            ctx = chunk[m["chunk_idx"]] if m["chunk_idx"] < len(chunk) else ""
            results.append({
                "score": float(sims[ti]),
                "doc": doc,
                "chunk": ctx,
            })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE TEXTO
# ──────────────────────────────────────────────────────────────────────────────

def pdf_to_text_and_preview(file) -> tuple[str, Image.Image | None, int]:
    """Devuelve (texto, imagen_previa, paginas). Requiere PyMuPDF."""
    if not fitz:
        return "", None, 0
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            textos = []
            preview_img = None
            for i, page in enumerate(doc):
                textos.append(page.get_text("text"))
                if i == 0:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    preview_img = img
            return "\n".join(textos), preview_img, len(doc)
    except Exception as e:
        return f"[ERROR al leer PDF] {e}", None, 0


def image_to_text(file) -> str:
    """OCR opcional mediante un modelo con visión vía OpenRouter; fallback: muestra nombre."""
    # Si hay un modelo con visión configurado, se envía la imagen como base64
    vision_model = st.session_state.settings.get("vision_model")
    key = get_openrouter_key()
    if key and vision_model:
        try:
            b = file.read()
            b64 = base64.b64encode(b).decode("utf-8")
            prompt = "Extrae texto legible del documento de la imagen. Responde solo con el texto plano."
            resp = openrouter_chat([
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_image", "image_data": b64},
                ]}
            ], model=vision_model)
            if resp:
                return resp
        except Exception as e:
            return f"[ERROR visión] {e}"
    # Fallback simple
    return "(OCR no configurado) Nombre de archivo: " + getattr(file, 'name', 'imagen')


# ──────────────────────────────────────────────────────────────────────────────
# OPENROUTER (LLM)
# ──────────────────────────────────────────────────────────────────────────────

def get_openrouter_key() -> str | None:
    return st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")


def openrouter_chat(messages: List[Dict[str, Any]], model: str | None = None, temperature: float = 0.1, max_tokens: int = 800) -> str:
    key = get_openrouter_key()
    if not key:
        st.warning("⚠️ Configura la clave de OpenRouter en Secrets o variable de entorno OPENROUTER_API_KEY.")
        return ""
    model = model or st.session_state.settings["model"]
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        # Estos headers son recomendados por OpenRouter para identificar tu app
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "RAG-Docs-Agenda",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"Error llamando al modelo: {e}")
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# LOGIN SENCILLO (usuario fijo: Carla)
# ──────────────────────────────────────────────────────────────────────────────

def login_box():
    st.sidebar.markdown("### 🔐 Acceso")
    with st.sidebar.form("login"):
        user = st.text_input("Usuario", value="Carla")
        pwd = st.text_input("Contraseña", type="password", help="Cambiar en Configuración.")
        ok = st.form_submit_button("Entrar")
    if ok:
        # Contraseña configurable por Secrets o variable ENTORNO; fallback genérico
        configured_pwd = st.secrets.get("APP_PASSWORD") or os.getenv("APP_PASSWORD") or "carla123"
        if user.strip().lower() == "carla" and pwd == configured_pwd:
            st.session_state.auth = True
            st.session_state.user = "Carla"
            log_event("login", "Acceso exitoso")
            st.success("Bienvenida, Carla ✨")
        else:
            st.session_state.auth = False
            st.error("Usuario o contraseña incorrectos.")


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR (CONFIGURACIÓN)
# ──────────────────────────────────────────────────────────────────────────────

def sidebar_config():
    st.sidebar.markdown("### ⚙️ Configuración")
    st.session_state.settings["project_name"] = st.sidebar.text_input("Nombre del proyecto", st.session_state.settings["project_name"])

    st.sidebar.markdown("**Modelo (OpenRouter)**")
    st.session_state.settings["model"] = st.sidebar.text_input("LLM texto", st.session_state.settings["model"], help="p. ej. openai/gpt-4o-mini, anthropic/claude-3.5-sonnet, meta/llama-3.1-70b-instruct")
    st.session_state.settings["vision_model"] = st.sidebar.text_input("LLM visión (opcional)", st.session_state.settings["vision_model"], help="para OCR vía modelo con visión")

    st.sidebar.divider()
    st.sidebar.markdown("**RAG**")
    st.session_state.settings["chunk_size"] = st.sidebar.slider("Tamaño de chunk", 400, 2000, st.session_state.settings["chunk_size"], step=50)
    st.session_state.settings["chunk_overlap"] = st.sidebar.slider("Solapamiento", 0, 400, st.session_state.settings["chunk_overlap"], step=20)
    st.session_state.settings["top_k"] = st.sidebar.slider("Top-K recuperación", 1, 10, st.session_state.settings["top_k"])

    st.sidebar.divider()
    st.sidebar.markdown("**Origen de documentos**")
    origen = st.sidebar.selectbox("Selecciona origen actual", ["Cargados manualmente", "Google Drive (próx)", "Correo (próx)", "S3 (próx)"])
    st.sidebar.caption("El origen se guardará como metadato por documento.")
    return origen


# ──────────────────────────────────────────────────────────────────────────────
# PERSISTENCIA LIGERA (export/import JSON del proyecto)
# ──────────────────────────────────────────────────────────────────────────────

def export_project_button():
    payload = {
        "settings": st.session_state.settings,
        "docs": st.session_state.docs,
        "events": st.session_state.events,
        "log": st.session_state.log,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }
    b = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("💾 Descargar proyecto (.json)", b, file_name=f"{st.session_state.settings['project_name']}.json", mime="application/json")


def import_project_uploader():
    up = st.file_uploader("Restaurar proyecto (.json)", type=["json"], accept_multiple_files=False)
    if up is not None:
        try:
            data = json.load(io.TextIOWrapper(up, encoding="utf-8"))
            st.session_state.settings = data.get("settings", st.session_state.settings)
            st.session_state.docs = data.get("docs", [])
            st.session_state.events = data.get("events", [])
            st.session_state.log = data.get("log", [])
            st.success("Proyecto restaurado.")
            log_event("import", f"Proyecto importado: {st.session_state.settings.get('project_name')}")
        except Exception as e:
            st.error(f"No se pudo importar: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# UI: CABECERA
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="app-title">📁 RAG de Documentos + 🗓️ Agenda</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Carga PDFs/Imágenes, extrae metadatos (fecha, folio, RFC, total), consulta con LLM (OpenRouter) y organiza actividades en un calendario y bitácora.</div>', unsafe_allow_html=True)

# Sidebar: login + config + persistencia
login_box()
if not st.session_state.auth:
    st.stop()

origen_actual = sidebar_config()
st.sidebar.divider()
export_project_button()
import_project_uploader()

# Métricas rápidas
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"<span class='metric-badge'>📄 Docs: <b>{len(st.session_state.docs)}</b></span>", unsafe_allow_html=True)
c2.markdown(f"<span class='metric-badge'>🧩 Chunks: <b>{(st.session_state.index['matrix'].shape[0] if st.session_state.index else 0)}</b></span>", unsafe_allow_html=True)
c3.markdown(f"<span class='metric-badge'>📅 Eventos: <b>{len(st.session_state.events)}</b></span>", unsafe_allow_html=True)
c4.markdown(f"<span class='metric-badge'>📝 Logs: <b>{len(st.session_state.log)}</b></span>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TABS PRINCIPALES
# ──────────────────────────────────────────────────────────────────────────────

t1, t2, t3, t4, t5, t6 = st.tabs([
    "📥 Carga & Extracción",
    "🔎 Buscador (RAG)",
    "🗓️ Calendario",
    "📓 Bitácora",
    "🧰 Admin/Config",
    "❔ Ayuda/Despliegue",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: CARGA & EXTRACCIÓN
# ──────────────────────────────────────────────────────────────────────────────
with t1:
    st.subheader("Carga de documentos")
    ups = st.file_uploader("Arrastra y suelta PDFs o imágenes (png, jpg, webp)", type=["pdf", "png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

    if ups:
        for f in ups:
            doc_id = make_id()
            texto, preview, paginas = "", None, 0
            if f.type == "application/pdf":
                texto, preview, paginas = pdf_to_text_and_preview(f)
            else:
                texto = image_to_text(f)
                try:
                    f.seek(0)
                    preview = Image.open(f).convert("RGB")
                except Exception:
                    preview = None

            md = extract_metadata(texto or "", f.name)
            doc = {
                "id": doc_id,
                "nombre": md.get("nombre") or f.name,
                "origen": origen_actual,
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
                "preview": None,  # no persistimos imagen en JSON export
            }
            st.session_state.docs.append(doc)
            log_event("doc:add", f"{f.name}", {"doc_id": doc_id, "origen": origen_actual, "paginas": paginas})

            with st.expander(f"📄 {f.name}"):
                cprev, cmeta = st.columns([1, 2])
                with cprev:
                    if preview is not None:
                        st.image(preview, caption=f"Vista previa ({paginas or 1} pág)", use_container_width=True)
                with cmeta:
                    st.write("**Metadatos detectados (editable)**")
                    editable = st.data_editor(
                        pd.DataFrame([{
                            "id": doc_id,
                            "Nombre": doc["nombre"],
                            "Fecha": doc["metadata"]["fecha"],
                            "Folio": doc["metadata"]["folio"],
                            "RFC": doc["metadata"]["rfc"],
                            "Total": doc["metadata"]["total"],
                            "Etiquetas": doc["metadata"]["etiquetas"],
                            "Origen": doc["origen"],
                        }]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    # guardar cambios
                    row = editable.iloc[0]
                    doc["nombre"] = row["Nombre"]
                    doc["origen"] = row["Origen"]
                    doc["metadata"].update({
                        "fecha": row["Fecha"],
                        "folio": row["Folio"],
                        "rfc": row["RFC"],
                        "total": row["Total"],
                        "etiquetas": row["Etiquetas"],
                    })
                    st.write("**Texto extraído (resumen)**")
                    st.text_area("", (texto or "")[:3000], height=200)

    st.divider()

    st.subheader("Tabla de documentos")
    if st.session_state.docs:
        df = pd.DataFrame([
            {
                "ID": d["id"],
                "Nombre": d["nombre"],
                "Origen": d["origen"],
                "Fecha": d["metadata"].get("fecha"),
                "Folio": d["metadata"].get("folio"),
                "RFC": d["metadata"].get("rfc"),
                "Total": d["metadata"].get("total"),
                "Páginas": d.get("paginas", 0),
                "Cargado": d.get("ts"),
                "Etiquetas": d["metadata"].get("etiquetas", ""),
            }
            for d in st.session_state.docs
        ])
        edited = st.data_editor(df, hide_index=True, use_container_width=True, num_rows="fixed")
        # Sincroniza cambios clave (nombre, origen, metadata)
        for i, row in edited.iterrows():
            d = next((x for x in st.session_state.docs if x["id"] == row["ID"]), None)
            if d:
                d["nombre"] = row["Nombre"]
                d["origen"] = row["Origen"]
                d["metadata"].update({
                    "fecha": row["Fecha"],
                    "folio": row["Folio"],
                    "rfc": row["RFC"],
                    "total": row["Total"],
                    "etiquetas": row.get("Etiquetas", ""),
                })

        csv = edited.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exportar docs (CSV)", csv, file_name="documentos.csv", mime="text/csv")

    else:
        st.info("No hay documentos. Carga algunos para comenzar.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: BUSCADOR (RAG) + QA
# ──────────────────────────────────────────────────────────────────────────────
with t2:
    st.subheader("Construir índice y consultar con LLM")
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("🧱 Reconstruir índice RAG", type="primary"):
            idx = build_index(st.session_state.docs, st.session_state.settings["chunk_size"], st.session_state.settings["chunk_overlap"])
            if idx:
                st.success(f"Índice construido con {idx['matrix'].shape[0]} chunks.")
                log_event("rag:index", f"chunks={idx['matrix'].shape[0]}")
            else:
                st.warning("No hay texto indexable.")

        st.markdown("**Parámetros**")
        st.write(st.session_state.settings)

    with c2:
        q = st.text_input("Pregunta sobre tus documentos", placeholder="¿Qué concepto total aparece en la factura de Juan del 3 de mayo?")
        if q:
            ctxs = retrieve(q, st.session_state.settings["top_k"]) if st.session_state.index else []
            if not ctxs:
                st.warning("Primero construye el índice o carga documentos.")
            else:
                # Prompt estructurado
                sys = (
                    "Eres un asistente que contesta SOLO usando el contexto proporcionado. "
                    "Entrega respuestas citando el nombre del documento y, si es posible, el folio o fecha. "
                    "Si no hay evidencia, responde que no está en los documentos."
                )
                context_blob = "\n\n".join([f"[Doc: {c['doc']['nombre']}]\n{c['chunk']}" for c in ctxs])
                messages = [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": f"Contexto:\n{context_blob}\n\nPregunta: {q}"},
                ]
                ans = openrouter_chat(messages, model=st.session_state.settings["model"], temperature=0.1, max_tokens=700)
                st.markdown("### Respuesta")
                st.write(ans)
                with st.expander("Contexto usado"):
                    for i, c in enumerate(ctxs, 1):
                        st.markdown(f"**{i}. {c['doc']['nombre']}** · score={c['score']:.3f}")
                        st.write(c["chunk"][:1500])
                log_event("rag:qa", f"Q={q}")

    st.divider()
    st.subheader("Resumen rápido de un documento con LLM")
    if st.session_state.docs:
        sel = st.selectbox("Selecciona documento", [f"{d['nombre']} ({d['id'][:6]})" for d in st.session_state.docs])
        d = next(x for x in st.session_state.docs if sel.startswith(x["nombre"]))
        if st.button("🧠 Resumir y extraer campos"):
            prompt = (
                "Lee el texto del documento y devuelve un JSON con: {nombre, fecha, folio, rfc, total, resumen}. "
                "Si algún campo no existe, pon null. Texto:\n\n" + (d.get("texto", "")[:12000])
            )
            out = openrouter_chat([
                {"role": "user", "content": prompt}
            ])
            st.write(out)
            log_event("rag:extract", f"doc={d['id']}")
    else:
        st.info("Carga documentos para habilitar el resumen.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: CALENDARIO
# ──────────────────────────────────────────────────────────────────────────────
with t3:
    st.subheader("Agenda y actividades")

    if calendar is None:
        st.info("Instala 'streamlit-calendar' para visualizar un calendario interactivo. Mientras tanto, usa el formulario.")
    else:
        cal_options = {
            "editable": True,
            "selectable": True,
            "initialView": "dayGridMonth",
            "locale": "es",
            "height": 700,
        }
        cal_events = st.session_state.events or []
        cal = calendar(events=cal_events, options=cal_options)
        if cal and cal.get("eventClick"):
            st.toast(f"Evento: {cal['eventClick']['event']['title']}")
        if cal and cal.get("selected"):
            sel = cal.get("selected")
            st.info(f"Seleccionado: {sel}")

    with st.form("add_event"):
        c1, c2 = st.columns(2)
        with c1:
            titulo = st.text_input("Título del evento", "Revisión de documentos")
            inicio = st.date_input("Fecha inicio", value=date.today())
            hora_ini = st.time_input("Hora inicio", value=datetime.now().time().replace(second=0, microsecond=0))
        with c2:
            fin = st.date_input("Fecha fin", value=date.today())
            hora_fin = st.time_input("Hora fin", value=(datetime.now() + timedelta(hours=1)).time().replace(second=0, microsecond=0))
            color = st.selectbox("Color", ["#3B82F6", "#22C55E", "#F59E0B", "#EF4444", "#8B5CF6"])  # azul, verde, ámbar, rojo, violeta
        notas = st.text_area("Notas")
        add_ok = st.form_submit_button("➕ Añadir al calendario")
        if add_ok:
            ev = {
                "title": titulo,
                "start": datetime.combine(inicio, hora_ini).isoformat(),
                "end": datetime.combine(fin, hora_fin).isoformat(),
                "color": color,
                "extendedProps": {"notas": notas, "creado_por": st.session_state.user},
            }
            st.session_state.events.append(ev)
            log_event("calendar:add", titulo)
            st.success("Evento agregado.")

    if st.session_state.events:
        st.markdown("**Próximos eventos**")
        df_ev = pd.DataFrame(st.session_state.events)
        st.dataframe(df_ev, use_container_width=True)
        st.download_button("📥 Exportar eventos (JSON)", json.dumps(st.session_state.events, ensure_ascii=False, indent=2), "eventos.json")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4: BITÁCORA
# ──────────────────────────────────────────────────────────────────────────────
with t4:
    st.subheader("Bitácora de actividades")
    if st.session_state.log:
        df_log = pd.DataFrame(st.session_state.log)
        tipo = st.selectbox("Filtrar por tipo", options=["(todos)"] + sorted(df_log.tipo.unique().tolist()))
        show = df_log if tipo == "(todos)" else df_log[df_log.tipo == tipo]
        st.dataframe(show.sort_values("timestamp", ascending=False), use_container_width=True)
        st.download_button("⬇️ Exportar bitácora (CSV)", show.to_csv(index=False).encode("utf-8"), "bitacora.csv")
    else:
        st.info("Aún no hay eventos en la bitácora.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5: ADMIN/CONFIG
# ──────────────────────────────────────────────────────────────────────────────
with t5:
    st.subheader("Ajustes avanzados")
    st.markdown("**Credenciales**")
    st.write("Para producción, agrega en *Secrets* (Streamlit Cloud) o variables de entorno:")
    st.code(
        """
        OPENROUTER_API_KEY = "sk-or-..."
        APP_PASSWORD = "cámbiame"
        """,
        language="bash",
    )

    st.markdown("**Limpieza**")
    c1, c2, c3 = st.columns(3)
    if c1.button("🧹 Limpiar documentos"):
        st.session_state.docs = []
        st.session_state.index = None
        log_event("admin:clear_docs", "")
        st.success("Documentos eliminados.")
    if c2.button("🧹 Limpiar bitácora"):
        st.session_state.log = []
        st.success("Bitácora limpia.")
    if c3.button("🧹 Limpiar eventos"):
        st.session_state.events = []
        st.success("Eventos eliminados.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 6: AYUDA / DESPLIEGUE
# ──────────────────────────────────────────────────────────────────────────────
with t6:
    st.subheader("Cómo desplegar en Streamlit Cloud (GitHub)")
    st.markdown(
        """
        1) Crea un repo en GitHub con `app.py`.
        2) Añade `requirements.txt` con:
           
           ```
           streamlit
           pandas
           numpy
           requests
           python-dateutil
           scikit-learn
           pillow
           pymupdf
           streamlit-calendar
           ```
           
        3) En **Streamlit Cloud** → **New app** → selecciona tu repo y rama, archivo principal `app.py`.
        4) En **Advanced settings → Secrets**, pega:
           
           ```
           OPENROUTER_API_KEY="sk-or-..."
           APP_PASSWORD="cámbiame"
           ```
           
        5) ¡Listo! Inicia la app y prueba con algunos PDFs o imágenes (para OCR avanzado usa un modelo de visión en OpenRouter y colócalo en Configuración).
        
        **Notas**
        - Este RAG usa TF‑IDF (ligero y sin costos). Puedes cambiar a embeddings externos si lo deseas.
        - Los metadatos (folio/RFC/fecha/total) se extraen con regex heurísticos; edítalos en la tabla.
        - El calendario es editable e incluye notas por evento.
        - La bitácora registra entradas de login, carga de docs, construcción del índice y consultas.
        - Seguridad básica para demo (usuario fijo *Carla*). Para producción considera OAuth o `streamlit-authenticator`.
        """
    )

# Fin del archivo
