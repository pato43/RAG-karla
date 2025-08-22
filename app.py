import os, io, json, base64, re
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

try:
    import fitz
except Exception:
    fitz = None

try:
    from streamlit_calendar import calendar
except Exception:
    calendar = None

try:
    import easyocr
except Exception:
    easyocr = None

st.set_page_config(page_title="RAG + Agenda", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
      :root{--bg:#0b1220;--panel:#111827;--muted:#9ca3af;--text:#e5e7eb;--primary:#60a5fa;--accent:#34d399;--border:#1f2937}
      html,body,.stApp{background:var(--bg); color:var(--text)}
      .app-header{background:linear-gradient(120deg,#0f172a,#0b1324 60%,#0e1726); border:1px solid var(--border); border-radius:16px; padding:18px 20px}
      .app-title{font-size:1.6rem; font-weight:800}
      .app-sub{color:var(--muted)}
      .metric-card{background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:12px}
      .metric-card h3{font-size:.85rem; color:#9ca3af; margin:0 0 6px}
      .metric-card .val{font-size:1.25rem; font-weight:800}
      .stTabs [data-baseweb="tab-list"]{gap:8px}
      .stTabs [data-baseweb="tab"]{padding:10px 16px; border-radius:10px; background:#0f172a; border:1px solid var(--border); color:var(--text)}
      .stTabs [aria-selected="true"]{background:#111827 !important; border-color:#1f2937 !important}
      .login-wrap{display:flex; align-items:center; justify-content:center; height:70vh}
      .login-card{width:420px; background:#0f172acc; backdrop-filter: blur(6px); border:1px solid var(--border); border-radius:16px; padding:22px}
      .login-title{font-size:1.2rem; font-weight:800}
      .stButton>button,.stDownloadButton>button{border-radius:12px}
      .calendar-card{background:var(--panel); border:1px solid var(--border); border-radius:16px; padding:8px}
    </style>
    """,
    unsafe_allow_html=True,
)

if "auth" not in st.session_state:
    st.session_state.auth = False
if "user" not in st.session_state:
    st.session_state.user = None
if "docs" not in st.session_state:
    st.session_state.docs: List[Dict[str, Any]] = []
if "index" not in st.session_state:
    st.session_state.index = None
if "events" not in st.session_state:
    st.session_state.events = []
if "log" not in st.session_state:
    st.session_state.log: List[Dict[str, Any]] = []
if "settings" not in st.session_state:
    st.session_state.settings = {
        "project_name": "RAG-Agenda",
        "model": "openai/gpt-4o-mini",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 5,
        "force_ocr": True,
        "origen": "Cargados manualmente",
    }

_OCR_READER = None

def get_easyocr_reader():
    global _OCR_READER
    if _OCR_READER is None and easyocr is not None:
        try:
            _OCR_READER = easyocr.Reader(["es","en"], gpu=False)
        except Exception:
            _OCR_READER = None
    return _OCR_READER

def log_event(tipo: str, detalle: str, extra: Dict[str, Any] | None=None):
    st.session_state.log.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tipo": tipo,
        "detalle": detalle,
        "extra": extra or {},
        "usuario": st.session_state.user,
    })

def make_id() -> str:
    return base64.urlsafe_b64encode(os.urandom(9)).decode("utf-8").rstrip("=")

DATE_REGEXES = [r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"]
UUID_REGEX = r"\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\b"
RFC_REGEX = r"\b([A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3})\b"
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

def ocr_image(img: Image.Image) -> str:
    r = get_easyocr_reader()
    if r is None:
        return "(OCR no disponible)"
    arr = np.array(img)
    res = r.readtext(arr, detail=0, paragraph=True)
    return "\n".join([t.strip() for t in res if t and t.strip()])

def pdf_bytes_to_text_and_preview(b: bytes, force_ocr: bool) -> tuple[str, Image.Image | None, int]:
    if not fitz:
        return "", None, 0
    textos, preview = [], None
    with fitz.open(stream=b, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            t = page.get_text("text")
            need = force_ocr or (not t or len(t.strip()) < 10)
            if need and easyocr is not None:
                pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                t = ocr_image(img)
                if i == 0:
                    preview = img
            else:
                if i == 0:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
                    preview = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            textos.append(t)
        return "\n".join(textos), preview, len(textos)

def image_bytes_to_text_and_preview(b: bytes) -> tuple[str, Image.Image | None]:
    try:
        img = Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return "", None
    return (ocr_image(img) if easyocr is not None else "(OCR no disponible)", img)

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = re.sub(r"\s+"," ", text).strip()
    if not text:
        return []
    out, s, L = [], 0, len(text)
    while s < L:
        e = min(L, s + chunk_size)
        sl = text[s:e]
        p = max(sl.rfind(". "), sl.rfind("\n"))
        if p > 100:
            e = s + p + 1
        out.append(text[s:e])
        if e >= L:
            break
        s = max(0, e - overlap)
    return out

def build_index(documents: List[Dict[str, Any]], chunk_size: int, overlap: int):
    corpus, mapping = [], []
    for d in documents:
        for i, ch in enumerate(chunk_text(d.get("texto",""), chunk_size, overlap)):
            corpus.append(ch)
            mapping.append({"doc_id": d["id"], "chunk_idx": i})
    if not corpus:
        return None
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
    mat = vec.fit_transform(corpus)
    st.session_state.index = {"vectorizer": vec, "matrix": mat, "mapping": mapping}
    return st.session_state.index

def retrieve(query: str, top_k: int, origin_filter: List[str] | None=None) -> List[Dict[str, Any]]:
    idx = st.session_state.index
    if not idx:
        return []
    qv = idx["vectorizer"].transform([query])
    sims = cosine_similarity(qv, idx["matrix"]).ravel()
    order = np.argsort(-sims)
    out = []
    for ti in order:
        m = idx["mapping"][ti]
        doc = next((d for d in st.session_state.docs if d["id"] == m["doc_id"]), None)
        if doc and (not origin_filter or doc.get("origen") in origin_filter):
            chs = chunk_text(doc.get("texto",""), st.session_state.settings["chunk_size"], st.session_state.settings["chunk_overlap"]) or [""]
            ctx = chs[m["chunk_idx"]] if m["chunk_idx"] < len(chs) else ""
            out.append({"score": float(sims[ti]), "doc": doc, "chunk": ctx})
            if len(out) >= top_k:
                break
    return out

def get_openrouter_key() -> str | None:
    return st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

def openrouter_chat(messages: List[Dict[str, Any]], model: str | None=None, temperature: float=0.1, max_tokens: int=700) -> str:
    key = get_openrouter_key()
    if not key:
        return ""
    model = model or st.session_state.settings["model"]
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "HTTP-Referer": "https://streamlit.io", "X-Title": "RAG-Agenda"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    except Exception:
        return ""

def login_screen():
    st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">🔐 Acceso</div>', unsafe_allow_html=True)
    user = st.text_input("Usuario", value="Karla")
    pwd = st.text_input("Contraseña", type="password")
    col1, col2 = st.columns([1,1])
    with col1:
        remember = st.checkbox("Recordarme")
    with col2:
        ok = st.button("Entrar", type="primary", use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)
    if ok:
        configured = st.secrets.get("APP_PASSWORD") or os.getenv("APP_PASSWORD") or "karla123"
        if user.strip().lower()=="karla" and pwd==configured:
            st.session_state.auth = True
            st.session_state.user = "Karla"
            if remember:
                st.session_state["remember_me"] = True
            log_event("login","ok")
            st.rerun()
        else:
            st.error("Usuario o contraseña incorrectos.")

def sidebar_config():
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")
        st.session_state.settings["project_name"] = st.text_input("Proyecto", st.session_state.settings["project_name"])
        st.session_state.settings["model"] = st.text_input("Modelo (OpenRouter)", st.session_state.settings["model"]) 
        st.session_state.settings["chunk_size"] = st.slider("Tamaño de chunk", 400, 2000, st.session_state.settings["chunk_size"], step=50)
        st.session_state.settings["chunk_overlap"] = st.slider("Solapamiento", 0, 400, st.session_state.settings["chunk_overlap"], step=20)
        st.session_state.settings["top_k"] = st.slider("Escala K (Top‑K)", 1, 10, st.session_state.settings["top_k"])
        opts = ["Cargados manualmente","Dependencia A","Dependencia B","Correo","S3","Otro…"]
        sel = st.selectbox("Dependencia / Origen actual", opts, index=0)
        free = st.text_input("Especificar", value="") if sel=="Otro…" else ""
        st.session_state.settings["origen"] = free or sel
        st.session_state.settings["force_ocr"] = st.toggle("Forzar OCR", value=st.session_state.settings["force_ocr"])
        st.divider()
        if st.button("Exportar proyecto"):
            payload = {"settings": st.session_state.settings, "docs": st.session_state.docs, "events": st.session_state.events, "log": st.session_state.log, "exported_at": datetime.now().isoformat(timespec="seconds")}
            st.download_button("Descargar .json", json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), file_name=f"{st.session_state.settings['project_name']}.json", mime="application/json")
        up = st.file_uploader("Restaurar .json", type=["json"], accept_multiple_files=False)
        if up is not None:
            try:
                data = json.load(io.TextIOWrapper(up, encoding="utf-8"))
                st.session_state.settings = data.get("settings", st.session_state.settings)
                st.session_state.docs = data.get("docs", [])
                st.session_state.events = data.get("events", [])
                st.session_state.log = data.get("log", [])
                st.success("Proyecto restaurado")
            except Exception as e:
                st.error(f"Error al importar: {e}")

def header():
    st.markdown('<div class="app-header"><div class="app-title">🧠 RAG de Documentos · 🗓️ Agenda</div><div class="app-sub">Tema oscuro. Login de Karla. OCR local. Calendario con notas. RAG con OpenRouter.</div></div>', unsafe_allow_html=True)

if not st.session_state.auth:
    login_screen()
    st.stop()

sidebar_config()
header()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><h3>Documentos</h3><div class="val">{len(st.session_state.docs)}</div></div>', unsafe_allow_html=True)
with c2:
    chunks = (st.session_state.index["matrix"].shape[0] if st.session_state.index else 0)
    st.markdown(f'<div class="metric-card"><h3>Chunks</h3><div class="val">{chunks}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><h3>Eventos</h3><div class="val">{len(st.session_state.events)}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><h3>Bitácora</h3><div class="val">{len(st.session_state.log)}</div></div>', unsafe_allow_html=True)

TAB_DOCS, TAB_RAG, TAB_CAL, TAB_LOG, TAB_ADMIN = st.tabs([
    "📥 Documentos",
    "🔎 RAG",
    "🗓️ Calendario",
    "📓 Bitácora",
    "🧰 Config",
])

with TAB_DOCS:
    st.subheader("Carga y extracción")
    colU, colO = st.columns([3,1])
    with colU:
        ups = st.file_uploader("Arrastra PDFs/Imágenes (png, jpg, webp)", type=["pdf","png","jpg","jpeg","webp"], accept_multiple_files=True)
    with colO:
        force = st.toggle("Forzar OCR en PDF", value=st.session_state.settings["force_ocr"])
    if ups:
        for f in ups:
            b = f.read()
            doc_id = make_id()
            if f.type == "application/pdf":
                texto, preview, paginas = pdf_bytes_to_text_and_preview(b, force or st.session_state.settings["force_ocr"])
            else:
                texto, preview = image_bytes_to_text_and_preview(b)
                paginas = 1
            md = extract_metadata(texto or "", f.name)
            doc = {"id": doc_id, "nombre": md.get("nombre") or f.name, "origen": st.session_state.settings.get("origen"), "texto": texto or "", "paginas": paginas, "metadata": {"fecha": md.get("fecha"), "folio": md.get("folio"), "rfc": md.get("rfc"), "total": md.get("total"), "etiquetas": ""}, "ts": datetime.now().isoformat(timespec="seconds")}
            st.session_state.docs.append(doc)
            log_event("doc:add", f.name, {"doc_id": doc_id, "origen": doc["origen"], "paginas": paginas})
            with st.expander(f"📄 {f.name}"):
                cprev, cmeta = st.columns([1,2])
                with cprev:
                    if preview is not None:
                        st.image(preview, caption=f"Vista previa ({paginas} pág)", use_container_width=True)
                with cmeta:
                    editable = st.data_editor(pd.DataFrame([{ "id": doc_id, "Nombre": doc["nombre"], "Fecha": doc["metadata"]["fecha"], "Folio": doc["metadata"]["folio"], "RFC": doc["metadata"]["rfc"], "Total": doc["metadata"]["total"], "Etiquetas": doc["metadata"]["etiquetas"], "Origen": doc["origen"] }]), use_container_width=True, hide_index=True)
                    row = editable.iloc[0]
                    doc["nombre"] = row["Nombre"]; doc["origen"] = row["Origen"]
                    doc["metadata"].update({"fecha": row["Fecha"], "folio": row["Folio"], "rfc": row["RFC"], "total": row["Total"], "etiquetas": row["Etiquetas"]})
                    st.text_area("Texto", (texto or "")[:4000], height=220)
    st.divider()
    st.subheader("Tabla de documentos")
    if st.session_state.docs:
        df = pd.DataFrame([{ "ID": d["id"], "Nombre": d["nombre"], "Origen": d["origen"], "Fecha": d["metadata"].get("fecha"), "Folio": d["metadata"].get("folio"), "RFC": d["metadata"].get("rfc"), "Total": d["metadata"].get("total"), "Páginas": d.get("paginas",0), "Cargado": d.get("ts"), "Etiquetas": d["metadata"].get("etiquetas","") } for d in st.session_state.docs])
        edited = st.data_editor(df, hide_index=True, use_container_width=True)
        for _, row in edited.iterrows():
            d = next((x for x in st.session_state.docs if x["id"]==row["ID"]), None)
            if d:
                d["nombre"]=row["Nombre"]; d["origen"]=row["Origen"]
                d["metadata"].update({"fecha": row["Fecha"], "folio": row["Folio"], "rfc": row["RFC"], "total": row["Total"], "etiquetas": row.get("Etiquetas","")})
        st.download_button("⬇️ Exportar CSV", edited.to_csv(index=False).encode("utf-8"), file_name="documentos.csv", mime="text/csv")
    else:
        st.info("Carga documentos para comenzar.")

with TAB_RAG:
    st.subheader("Buscador + QA")
    cA, cB = st.columns([1,2])
    with cA:
        if st.button("🧱 (Re)construir índice", type="primary"):
            idx = build_index(st.session_state.docs, st.session_state.settings["chunk_size"], st.session_state.settings["chunk_overlap"])
            if idx:
                st.success(f"Índice listo con {idx['matrix'].shape[0]} chunks.")
            else:
                st.warning("No hay texto indexable.")
        st.caption("Ajusta Escala K en Config.")
    with cB:
        colq1, colq2 = st.columns([2,1])
        with colq1:
            q = st.text_input("Pregunta", placeholder="¿Cuál es el total de la factura X?")
        with colq2:
            filtros = sorted(set(d.get("origen") for d in st.session_state.docs))
            f_origen = st.multiselect("Dependencia/Origen", filtros)
        if q:
            ctxs = retrieve(q, st.session_state.settings["top_k"], origin_filter=f_origen if f_origen else None) if st.session_state.index else []
            if not ctxs:
                st.warning("Construye el índice o carga documentos.")
            else:
                blob = "\n\n".join([f"[Doc: {c['doc']['nombre']} | Fecha: {c['doc']['metadata'].get('fecha')} | Folio: {c['doc']['metadata'].get('folio')}]\n{c['chunk']}" for c in ctxs])
                msgs = [{"role":"system","content":"Responde usando SOLO el contexto. Cita Nombre/Fecha/Folio si aparece. Si falta evidencia, indícalo."},{"role":"user","content":f"Contexto:\n{blob}\n\nPregunta: {q}"}]
                ans = openrouter_chat(msgs, model=st.session_state.settings["model"], temperature=0.1, max_tokens=700)
                st.markdown("### Respuesta")
                st.write(ans or "(sin respuesta)")
                pack = {"question": q, "answer": ans, "used": [{"doc": c["doc"]["nombre"], "fecha": c["doc"]["metadata"].get("fecha"), "folio": c["doc"]["metadata"].get("folio"), "score": c["score"]} for c in ctxs]}
                st.download_button("💾 Guardar respuesta", json.dumps(pack, ensure_ascii=False, indent=2), "respuesta.json")
                with st.expander("Contexto usado"):
                    for i, c in enumerate(ctxs,1):
                        meta=c['doc']['metadata']
                        st.markdown(f"**{i}. {c['doc']['nombre']}** · score={c['score']:.3f} · Fecha: {meta.get('fecha')} · Folio: {meta.get('folio')}")
                        st.write(c["chunk"][:1600])

with TAB_CAL:
    st.subheader("Agenda y actividades")
    if calendar is None:
        st.info("Instala 'streamlit-calendar' para ver el calendario interactivo.")
    else:
        cal_opts = {"editable": True, "selectable": True, "initialView": "dayGridMonth", "locale": "es", "height": 760}
        st.markdown('<div class="calendar-card">', unsafe_allow_html=True)
        cal = calendar(events=st.session_state.events or [], options=cal_opts)
        st.markdown('</div>', unsafe_allow_html=True)
        if cal and cal.get("eventClick"):
            st.toast(f"Evento: {cal['eventClick']['event']['title']}")
    with st.form("add_event"):
        c1, c2 = st.columns(2)
        with c1:
            titulo = st.text_input("Título", "Revisión de documentos")
            inicio = st.date_input("Fecha inicio", value=date.today())
            hora_ini = st.time_input("Hora inicio", value=datetime.now().time().replace(second=0, microsecond=0))
        with c2:
            fin = st.date_input("Fecha fin", value=date.today())
            hora_fin = st.time_input("Hora fin", value=(datetime.now()+timedelta(hours=1)).time().replace(second=0, microsecond=0))
            color = st.selectbox("Color", ["#60a5fa","#34d399","#f59e0b","#ef4444","#8b5cf6"])
        notas = st.text_area("Notas")
        if st.form_submit_button("➕ Añadir"):
            ev = {"title": titulo, "start": datetime.combine(inicio, hora_ini).isoformat(), "end": datetime.combine(fin, hora_fin).isoformat(), "color": color, "extendedProps": {"notas": notas, "creado_por": st.session_state.user}}
            st.session_state.events.append(ev)
            log_event("calendar:add", titulo)
            st.success("Evento agregado.")
    if st.session_state.events:
        st.markdown("**Próximos eventos**")
        df_ev = pd.DataFrame(st.session_state.events)
        st.dataframe(df_ev, use_container_width=True)
        st.download_button("📥 Exportar eventos", json.dumps(st.session_state.events, ensure_ascii=False, indent=2), "eventos.json")

with TAB_LOG:
    st.subheader("Bitácora")
    if st.session_state.log:
        df_log = pd.DataFrame(st.session_state.log)
        tipo = st.selectbox("Filtrar", options=["(todos)"] + sorted(df_log.tipo.unique().tolist()))
        show = df_log if tipo == "(todos)" else df_log[df_log.tipo == tipo]
        st.dataframe(show.sort_values("timestamp", ascending=False), use_container_width=True)
        st.download_button("⬇️ Exportar", show.to_csv(index=False).encode("utf-8"), "bitacora.csv")
    else:
        st.info("Aún no hay eventos.")

with TAB_ADMIN:
    st.subheader("Mantenimiento")
    col1, col2, col3 = st.columns(3)
    if col1.button("🧹 Limpiar documentos"):
        st.session_state.docs = []
        st.session_state.index = None
        log_event("admin:clear_docs", "")
        st.success("Documentos eliminados.")
    if col2.button("🧹 Limpiar bitácora"):
        st.session_state.log = []
        st.success("Bitácora limpia.")
    if col3.button("🧹 Limpiar eventos"):
        st.session_state.events = []
        st.success("Eventos eliminados.")
