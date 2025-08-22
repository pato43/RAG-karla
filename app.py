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
from PIL import Image, ImageDraw, ImageFont

# Opcionales
try: import fitz  # PyMuPDF
except Exception: fitz = None

try: from streamlit_calendar import calendar
except Exception: calendar = None

try: import easyocr
except Exception: easyocr = None

# ── UI base (oscuro)
st.set_page_config(page_title="RAG + Agenda", page_icon="🧠", layout="wide")
st.markdown("""
<style>
  :root{--bg:#0b1220;--panel:#0f172a;--muted:#94a3b8;--text:#e5e7eb;--primary:#60a5fa;--accent:#34d399;--border:#1e293b}
  html,body,.stApp{background:var(--bg); color:var(--text)}
  .hdr{background:#0c1423;border:1px solid var(--border);border-radius:16px;padding:16px 20px;margin-bottom:12px}
  .t1{font-size:1.7rem;font-weight:800}.t2{color:var(--muted)}
  .metric{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:12px}
  .metric h3{font-size:.85rem;color:#93a4b2;margin:0 0 6px}.metric .v{font-weight:800}
  .stTabs [data-baseweb="tab-list"]{gap:8px}
  .stTabs [data-baseweb="tab"]{padding:10px 16px;border-radius:10px;background:#0d1627;border:1px solid var(--border);color:var(--text)}
  .stTabs [aria-selected="true"]{background:#0f192b!important;border-color:#203049!important}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:12px}
  .pill{display:inline-block;padding:2px 8px;border:1px solid var(--border);background:#0c1423;border-radius:999px;font-size:12px;color:#cbd5e1}
  .login-wrap{display:flex;align-items:center;justify-content:center;height:78vh}
  .login-card{width:420px;background:#0f172acc;border:1px solid var(--border);border-radius:16px;padding:22px}
  .login-title{font-size:1.2rem;font-weight:800;margin-bottom:6px}
  .calendar-card{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:8px}
  .stButton>button,.stDownloadButton>button{border-radius:12px}
</style>
""", unsafe_allow_html=True)

# ── Estado
if "auth" not in st.session_state: st.session_state.auth = False
if "user" not in st.session_state: st.session_state.user = None
if "docs" not in st.session_state: st.session_state.docs: List[Dict[str, Any]] = []
if "index" not in st.session_state: st.session_state.index = None
if "events" not in st.session_state: st.session_state.events = []
if "log" not in st.session_state: st.session_state.log: List[Dict[str, Any]] = []
if "settings" not in st.session_state:
    st.session_state.settings = {
        "project_name": "RAG-Agenda",
        "model": "openai/gpt-4o-mini",
        "vision_model": "openai/gpt-4o-mini",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 5,
        "force_ocr": True,
        "origen": "Cargados manualmente",
    }

# ── Utils
def log_event(tipo: str, detalle: str, extra: Dict[str, Any] | None=None):
    st.session_state.log.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tipo": tipo, "detalle": detalle, "extra": extra or {}, "usuario": st.session_state.user
    })

def make_id() -> str:
    return base64.urlsafe_b64encode(os.urandom(9)).decode("utf-8").rstrip("=")

DATE_REGEXES=[r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"]
UUID_REGEX=r"\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\b"
RFC_REGEX=r"\b([A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3})\b"
TOTAL_REGEX=r"(?i)\b(total|importe|monto)\b\D{0,15}([\$\s]*[0-9]+[\,\.]?[0-9]{0,2})"
FOLIO_HINTS=r"(?i)(folio|factura|no\.?\s*\w*|id|referencia)\s*[:#\-]?\s*([A-Za-z0-9\-]{5,})"

def parse_dates(text: str) -> str | None:
    for rgx in DATE_REGEXES:
        m = re.search(rgx, text)
        if m:
            try: return dateparser.parse(m.group(1), dayfirst=True, fuzzy=True).date().isoformat()
            except Exception: pass
    return None

def extract_metadata(text: str, filename: str) -> Dict[str, Any]:
    md = {"nombre": filename, "fecha": parse_dates(text) or None, "folio": None, "rfc": None, "total": None, "etiquetas": ""}
    m = re.search(UUID_REGEX, text)
    if m: md["folio"] = m.group(0)
    else:
        m2 = re.search(FOLIO_HINTS, text)
        if m2: md["folio"] = m2.group(2)[:40]
    m = re.search(RFC_REGEX, text)
    if m: md["rfc"] = m.group(1)
    m = re.search(TOTAL_REGEX, text)
    if m: md["total"] = m.group(2).strip()
    return md

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text: return []
    out, s, L = [], 0, len(text)
    while s < L:
        e = min(L, s + chunk_size)
        sl = text[s:e]; p = max(sl.rfind(". "), sl.rfind("\n"))
        if p > 100: e = s + p + 1
        out.append(text[s:e])
        if e >= L: break
        s = max(0, e - overlap)
    return out

def build_index(documents: List[Dict[str, Any]], chunk_size: int, overlap: int):
    corpus, mapping = [], []
    for d in documents:
        for i, ch in enumerate(chunk_text(d.get("texto",""), chunk_size, overlap)):
            corpus.append(ch); mapping.append({"doc_id": d["id"], "chunk_idx": i})
    if not corpus: return None
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
    mat = vec.fit_transform(corpus)
    st.session_state.index = {"vectorizer": vec, "matrix": mat, "mapping": mapping}
    return st.session_state.index

def retrieve(query: str, top_k: int, origin_filter: List[str] | None=None) -> List[Dict[str, Any]]:
    idx = st.session_state.index
    if not idx: return []
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
            if len(out) >= top_k: break
    return out

# ── OCR + previews
_OCR = None
def ocr_reader():
    global _OCR
    if _OCR is None and easyocr is not None:
        try: _OCR = easyocr.Reader(["es","en"], gpu=False)
        except Exception: _OCR = None
    return _OCR

def ocr_image_text(img: Image.Image) -> str:
    r = ocr_reader()
    if r is None: return ""
    arr = np.array(img)
    try:
        res = r.readtext(arr, detail=0, paragraph=True)
        return "\n".join([t.strip() for t in res if t and t.strip()])
    except Exception: return ""

def annotate_preview(img: Image.Image, text: str, max_chars: int=260) -> Image.Image | None:
    if img is None: return None
    W, H = img.size
    box_h = int(H*0.22)
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay, 'RGBA')
    draw.rectangle([(0,H-box_h),(W,H)], fill=(12,20,35,210))
    try: font = ImageFont.truetype("DejaVuSans.ttf", size=max(16,int(W*0.018)))
    except Exception: font = ImageFont.load_default()
    pad = 18
    t = (text or "").replace("\r"," ").replace("\n"," ")
    t = t[:max_chars] + ("…" if len(t)>max_chars else "")
    draw.multiline_text((pad, H-box_h+pad), t, fill=(226,232,240,255), font=font, spacing=4)
    return overlay

def pdf_bytes_to_text_and_preview(b: bytes, force_ocr: bool) -> tuple[str, Image.Image | None, int]:
    if not fitz: return "", None, 0
    textos, preview = [], None
    with fitz.open(stream=b, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            t = page.get_text("text") or ""
            need = force_ocr or (not t or len(t.strip()) < 10)
            if need:
                pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                t_ocr = ocr_image_text(img) if easyocr is not None else ""
                t = t_ocr or t
                if i == 0: preview = img
            else:
                if i == 0:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
                    preview = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            textos.append(t)
        return "\n".join(textos), preview, len(doc)

def image_bytes_to_text_and_image(b: bytes) -> tuple[str, Image.Image | None]:
    try:
        img = Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return "", None
    txt = ocr_image_text(img) if easyocr is not None else ""
    if not txt:
        key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        vm = st.session_state.settings.get("vision_model")
        if key and vm:
            try:
                b64 = base64.b64encode(b).decode("utf-8")
                prompt = "Extrae texto legible del documento de la imagen. Devuelve solo texto plano."
                txt = openrouter_chat([{"role":"user","content":[{"type":"text","text":prompt},{"type":"input_image","image_data":b64}]}], model=vm)
            except Exception: pass
    return txt or "", img

# ── OpenRouter
def get_openrouter_key() -> str | None:
    return st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

def openrouter_chat(messages: List[Dict[str, Any]], model: str | None=None, temperature: float=0.1, max_tokens: int=700) -> str:
    key = get_openrouter_key()
    if not key: return ""
    model = model or st.session_state.settings["model"]
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://streamlit.io", "X-Title": "RAG-Agenda"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    except Exception: return ""

# ── Login (Karla)
def login_screen():
    st.markdown('<div class="login-wrap"><div class="login-card"><div class="login-title">🔐 Acceso</div>', unsafe_allow_html=True)
    u = st.text_input("Usuario", value="Karla")
    p = st.text_input("Contraseña", type="password")
    ok = st.button("Entrar", type="primary", use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)
    if ok:
        configured = st.secrets.get("APP_PASSWORD") or os.getenv("APP_PASSWORD") or "karla123"
        if u.strip().lower()=="karla" and p==configured:
            st.session_state.auth=True; st.session_state.user="Karla"; log_event("login","ok"); st.rerun()
        else: st.error("Usuario o contraseña incorrectos.")

# ── Sidebar (con ayuda)
def sidebar_config():
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")
        st.session_state.settings["project_name"] = st.text_input("Proyecto", st.session_state.settings["project_name"])
        st.session_state.settings["model"] = st.text_input("Modelo (OpenRouter)", st.session_state.settings["model"])
        st.session_state.settings["vision_model"] = st.text_input("Modelo visión (opcional)", st.session_state.settings["vision_model"])
        st.session_state.settings["chunk_size"] = st.slider("Tamaño de chunk", 400, 2000, st.session_state.settings["chunk_size"], step=50)
        st.session_state.settings["chunk_overlap"] = st.slider("Solapamiento", 0, 400, st.session_state.settings["chunk_overlap"], step=20)
        st.session_state.settings["top_k"] = st.slider("Top-K (K)", 1, 10, st.session_state.settings["top_k"])
        st.session_state.settings["force_ocr"] = st.toggle("Forzar OCR en PDFs", value=st.session_state.settings["force_ocr"])
        opts = ["Cargados manualmente","Dependencia A","Dependencia B","Correo","S3","Otro…"]
        sel = st.selectbox("Dependencia/Origen", opts, index=0)
        st.session_state.settings["origen"] = st.text_input("Especificar", "") if sel=="Otro…" else sel

        with st.expander("ℹ️ ¿Qué hace cada control?"):
            st.markdown("""
- **Proyecto**: nombre para exportar/importar tus datos.
- **Modelo (OpenRouter)**: LLM de texto para preguntas (ej. `openai/gpt-4o-mini`).
- **Modelo visión**: opcional; si no hay EasyOCR, usa este LLM para leer imágenes.
- **Tamaño de chunk / Solapamiento**: cómo troceamos el texto para recuperar contexto.
- **Top-K**: cuántos trozos recuperamos para responder.
- **Forzar OCR en PDFs**: rasteriza y aplica OCR cuando el PDF viene como imagen.
- **Dependencia/Origen**: etiqueta de procedencia que se guarda en cada documento.
            """)

        st.divider()
        payload = {"settings": st.session_state.settings, "docs": st.session_state.docs,
                   "events": st.session_state.events, "log": st.session_state.log,
                   "exported_at": datetime.now().isoformat(timespec="seconds")}
        st.download_button("💾 Exportar proyecto (.json)", json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                           file_name=f"{st.session_state.settings['project_name']}.json", mime="application/json")
        up = st.file_uploader("Restaurar proyecto (.json)", type=["json"], accept_multiple_files=False)
        if up is not None:
            try:
                data = json.load(io.TextIOWrapper(up, encoding="utf-8"))
                st.session_state.settings = data.get("settings", st.session_state.settings)
                st.session_state.docs = data.get("docs", [])
                st.session_state.events = data.get("events", [])
                st.session_state.log = data.get("log", [])
                st.success("Proyecto restaurado.")
            except Exception as e: st.error(f"Error al importar: {e}")

# ── Header
st.markdown('<div class="hdr"><div class="t1">🧠 RAG de Documentos · 🗓️ Agenda</div><div class="t2">OCR (PDF+img) + Texto general + RAG con OpenRouter + Calendario + Bitácora</div></div>', unsafe_allow_html=True)

if not st.session_state.auth:
    login_screen()
    st.stop()

sidebar_config()

# Métricas
c1,c2,c3,c4 = st.columns(4)
with c1: st.markdown(f'<div class="metric"><h3>Documentos</h3><div class="v">{len(st.session_state.docs)}</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric"><h3>Chunks</h3><div class="v">{(st.session_state.index["matrix"].shape[0] if st.session_state.index else 0)}</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric"><h3>Eventos</h3><div class="v">{len(st.session_state.events)}</div></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric"><h3>Bitácora</h3><div class="v">{len(st.session_state.log)}</div></div>', unsafe_allow_html=True)

# Tabs
TAB_DOCS, TAB_TEXT, TAB_RAG, TAB_CAL, TAB_LOG, TAB_ADMIN = st.tabs(
    ["📥 Documentos","🧾 Texto","🔎 RAG","🗓️ Calendario","📓 Bitácora","🧰 Admin"]
)

# ── Documentos (carga + metadatos + preview anotado)
with TAB_DOCS:
    st.subheader("Carga y extracción")
    ups = st.file_uploader("Arrastra PDFs/Imágenes (png, jpg, webp)", type=["pdf","png","jpg","jpeg","webp"], accept_multiple_files=True)
    if ups:
        for f in ups:
            b = f.read()
            doc_id = make_id()
            if f.type == "application/pdf":
                texto, preview, paginas = pdf_bytes_to_text_and_preview(b, st.session_state.settings["force_ocr"])
            else:
                texto, preview = image_bytes_to_text_and_image(b); paginas = 1
            md = extract_metadata(texto or "", f.name)
            doc = {"id":doc_id,"nombre":md.get("nombre") or f.name,"origen":st.session_state.settings.get("origen"),
                   "texto":texto or "","paginas":paginas,"metadata":{"fecha":md.get("fecha"),"folio":md.get("folio"),
                   "rfc":md.get("rfc"),"total":md.get("total"),"etiquetas":""},"ts":datetime.now().isoformat(timespec="seconds")}
            st.session_state.docs.append(doc); log_event("doc:add", f.name, {"doc_id":doc_id,"origen":doc["origen"],"paginas":paginas})

            with st.expander(f"📄 {f.name}"):
                cprev, cmeta = st.columns([1,2])
                with cprev:
                    if preview is not None:
                        ann = annotate_preview(preview, texto)
                        st.image(ann or preview, caption=f"Vista previa anotada ({paginas or 1} pág)", use_container_width=True)
                        buf=io.BytesIO(); (ann or preview).save(buf, format="PNG")
                        st.download_button("⬇️ PNG anotado", buf.getvalue(), file_name=f"{os.path.splitext(f.name)[0]}_anotado.png", mime="image/png")
                        st.download_button("⬇️ Texto (.txt)", (texto or "").encode("utf-8"), file_name=f"{os.path.splitext(f.name)[0]}.txt", mime="text/plain")
                with cmeta:
                    editable = st.data_editor(pd.DataFrame([{
                        "id":doc_id,"Nombre":doc["nombre"],"Fecha":doc["metadata"]["fecha"],"Folio":doc["metadata"]["folio"],
                        "RFC":doc["metadata"]["rfc"],"Total":doc["metadata"]["total"],"Etiquetas":doc["metadata"]["etiquetas"],"Origen":doc["origen"]
                    }]), use_container_width=True, hide_index=True)
                    row = editable.iloc[0]
                    doc["nombre"]=row["Nombre"]; doc["origen"]=row["Origen"]
                    doc["metadata"].update({"fecha":row["Fecha"],"folio":row["Folio"],"rfc":row["RFC"],"total":row["Total"],"etiquetas":row["Etiquetas"]})
                    st.text_area("Texto extraído (resumen)", (texto or "")[:2000], height=180)

    st.divider()
    st.subheader("Tabla de documentos")
    if st.session_state.docs:
        df = pd.DataFrame([{ "ID":d["id"],"Nombre":d["nombre"],"Origen":d["origen"],"Fecha":d["metadata"].get("fecha"),
            "Folio":d["metadata"].get("folio"),"RFC":d["metadata"].get("rfc"),"Total":d["metadata"].get("total"),
            "Páginas":d.get("paginas",0),"Cargado":d.get("ts"),"Etiquetas":d["metadata"].get("etiquetas","") }
            for d in st.session_state.docs])
        edited = st.data_editor(df, hide_index=True, use_container_width=True)
        for _, row in edited.iterrows():
            d = next((x for x in st.session_state.docs if x["id"]==row["ID"]), None)
            if d:
                d["nombre"]=row["Nombre"]; d["origen"]=row["Origen"]
                d["metadata"].update({"fecha":row["Fecha"],"folio":row["Folio"],"rfc":row["RFC"],"total":row["Total"],"etiquetas":row.get("Etiquetas","")})
        st.download_button("⬇️ Exportar CSV", edited.to_csv(index=False).encode("utf-8"), file_name="documentos.csv", mime="text/csv")
    else:
        st.info("Carga documentos para comenzar.")

# ── TEXTO GENERAL (acceso completo + buscar + exportar)
with TAB_TEXT:
    st.subheader("Texto general del documento")
    if not st.session_state.docs:
        st.info("Carga documentos para ver el texto general.")
    else:
        options = [f"{d['nombre']} — {d['id'][:6]}" for d in st.session_state.docs]
        sel = st.selectbox("Documento", options)
        dsel = next(d for d in st.session_state.docs if sel.startswith(d["nombre"]))
        colA, colB = st.columns([3,1])
        with colA:
            st.text_area("Texto (completo)", dsel.get("texto",""), height=520)
        with colB:
            st.write("**Datos**")
            st.json({"Nombre": dsel["nombre"], **dsel["metadata"], "Páginas": dsel.get("paginas",0)}, expanded=False)
            st.download_button("⬇️ Descargar texto (.txt)", (dsel.get("texto","")).encode("utf-8"),
                               file_name=f"{os.path.splitext(dsel['nombre'])[0]}.txt", mime="text/plain")
            st.write("**Buscar**")
            q = st.text_input("Palabra/frase")
            if q:
                txt = dsel.get("texto","")
                matches = [m.start() for m in re.finditer(re.escape(q), txt, flags=re.IGNORECASE)]
                st.caption(f"Coincidencias: {len(matches)}")
                for i, pos in enumerate(matches[:10], 1):
                    s = max(0, pos-80); e = min(len(txt), pos+80)
                    st.markdown(f"**{i}.** …{txt[s:e]}…")

# ── RAG
with TAB_RAG:
    st.subheader("Buscador + QA")
    cA, cB = st.columns([1,2])
    with cA:
        if st.button("🧱 (Re)construir índice", type="primary"):
            idx = build_index(st.session_state.docs, st.session_state.settings["chunk_size"], st.session_state.settings["chunk_overlap"])
            if idx: st.success(f"Índice listo con {idx['matrix'].shape[0]} chunks.")
            else: st.warning("No hay texto indexable.")
        filtros = sorted(set(d.get("origen") for d in st.session_state.docs))
        f_origen = st.multiselect("Filtrar por origen", filtros)
        st.caption("Ajusta K en Configuración (barra lateral).")
    with cB:
        q = st.text_input("Pregunta", placeholder="¿Cuál es el total de la factura X?")
        if q:
            ctxs = retrieve(q, st.session_state.settings["top_k"], origin_filter=f_origen if f_origen else None) if st.session_state.index else []
            if not ctxs: st.warning("Construye el índice o carga documentos.")
            else:
                blob = "\n\n".join([f"[Doc: {c['doc']['nombre']} | Fecha: {c['doc']['metadata'].get('fecha')} | Folio: {c['doc']['metadata'].get('folio')}]\n{c['chunk']}" for c in ctxs])
                msgs = [
                    {"role":"system","content":"Responde usando SOLO el contexto. Cita Nombre/Fecha/Folio si aparece. Si falta evidencia, dilo."},
                    {"role":"user","content":f"Contexto:\n{blob}\n\nPregunta: {q}"},
                ]
                ans = openrouter_chat(msgs, model=st.session_state.settings["model"], temperature=0.1, max_tokens=700)
                st.markdown("### Respuesta"); st.write(ans or "(sin respuesta)")
                with st.expander("Contexto usado"):
                    for i, c in enumerate(ctxs,1):
                        meta=c['doc']['metadata']
                        st.markdown(f"**{i}. {c['doc']['nombre']}** · score={c['score']:.3f} · Fecha: {meta.get('fecha')} · Folio: {meta.get('folio')}")
                        st.write(c["chunk"][:1600])

# ── Calendario (interactivo)
with TAB_CAL:
    st.subheader("Agenda y actividades")
    colL, colR = st.columns([2,1])
    with colL:
        if calendar is None:
            st.info("Instala 'streamlit-calendar' para ver el calendario interactivo.")
        else:
            options = {
                "editable": True, "selectable": True, "initialView": "dayGridMonth", "locale": "es", "height": 760,
                "headerToolbar": {"left":"prev,next today","center":"title","right":"dayGridMonth,timeGridWeek,timeGridDay,listWeek"},
            }
            st.markdown('<div class="calendar-card">', unsafe_allow_html=True)
            cal = calendar(events=st.session_state.events or [], options=options)
            st.markdown('</div>', unsafe_allow_html=True)

            # Persistencia de interacciones (si la lib las expone)
            if cal:
                if cal.get("eventClick"):
                    ev = cal["eventClick"]["event"]; st.toast(f"{ev['title']} — {ev['start']}")
                if cal.get("eventAdd"):  # nuevo evento desde calendar UI
                    ev = cal["eventAdd"]["event"]; st.session_state.events.append(ev); st.success("Evento agregado desde calendario.")
                if cal.get("eventChange"):  # drag/resize
                    ev = cal["eventChange"]["event"]
                    for e in st.session_state.events:
                        if e.get("id", e.get("title")) == ev.get("id", ev.get("title")):
                            e.update(ev); break
                if cal.get("eventRemove"):
                    ev = cal["eventRemove"]["event"]
                    st.session_state.events = [e for e in st.session_state.events if e.get("id", e.get("title")) != ev.get("id", ev.get("title"))]

    with colR:
        with st.form("add_event"):
            titulo = st.text_input("Título","Revisión de documentos")
            inicio = st.date_input("Inicio", value=date.today())
            hora_ini = st.time_input("Hora inicio", value=datetime.now().time().replace(second=0, microsecond=0))
            fin = st.date_input("Fin", value=date.today())
            hora_fin = st.time_input("Hora fin", value=(datetime.now()+timedelta(hours=1)).time().replace(second=0, microsecond=0))
            color = st.selectbox("Color", ["#60a5fa","#34d399","#f59e0b","#ef4444","#8b5cf6"])
            notas = st.text_area("Notas")
            if st.form_submit_button("➕ Añadir"):
                ev = {"title": titulo, "start": datetime.combine(inicio,hora_ini).isoformat(),
                      "end": datetime.combine(fin,hora_fin).isoformat(), "color": color,
                      "extendedProps": {"notas": notas, "creado_por": st.session_state.user}}
                st.session_state.events.append(ev); log_event("calendar:add", titulo); st.success("Evento agregado.")
        if st.session_state.events:
            st.markdown("#### Próximos 7 días")
            def parse_iso(ts):
                try: return datetime.fromisoformat(ts)
                except Exception: return None
            up=[(parse_iso(e["start"]), e) for e in st.session_state.events if e.get("start")]
            up=[x for x in up if x[0] is not None]; up=sorted(up,key=lambda x:x[0])
            now=datetime.now(); future=[e for t,e in up if 0 <= (t-now).days <= 7][:8]
            for e in future:
                st.markdown(f"<div class='card'><b>{e['title']}</b><br><span class='pill'>{e['start']}</span></div>", unsafe_allow_html=True)

# ── Bitácora
with TAB_LOG:
    st.subheader("Bitácora")
    if st.session_state.log:
        df=pd.DataFrame(st.session_state.log)
        tipo=st.selectbox("Filtrar", options=["(todos)"]+sorted(df.tipo.unique().tolist()))
        show=df if tipo=="(todos)" else df[df.tipo==tipo]
        st.dataframe(show.sort_values("timestamp", ascending=False), use_container_width=True)
        st.download_button("⬇️ Exportar", show.to_csv(index=False).encode("utf-8"), "bitacora.csv")
    else: st.info("Aún no hay eventos.")

# ── Admin
with TAB_ADMIN:
    st.subheader("Mantenimiento")
    c1,c2,c3=st.columns(3)
    if c1.button("🧹 Limpiar documentos"): st.session_state.docs=[]; st.session_state.index=None; log_event("admin:clear_docs",""); st.success("Documentos eliminados.")
    if c2.button("🧹 Limpiar bitácora"): st.session_state.log=[]; st.success("Bitácora limpia.")
    if c3.button("🧹 Limpiar eventos"): st.session_state.events=[]; st.success("Eventos eliminados.")
