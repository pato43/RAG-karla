# app.py — RAG + Agenda (oscuro, Karla, OCR mejorado + LLM visión, calendario estético)

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
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

try: import fitz  # PyMuPDF
except Exception: fitz = None

try: from streamlit_calendar import calendar
except Exception: calendar = None

try: import easyocr
except Exception: easyocr = None

st.set_page_config(page_title="RAG + Agenda", page_icon="🧠", layout="wide")

st.markdown("""
<style>
  :root{--bg:#0b1220;--panel:#0f172a;--muted:#94a3b8;--text:#e5e7eb;--primary:#60a5fa;--accent:#34d399;--border:#1e293b}
  html,body,.stApp{background:radial-gradient(1400px 700px at 15% -10%,#0e1628 0%,#0b1220 55%,#0a0f1a 100%); color:var(--text)}
  .hdr{background:#0c1423;border:1px solid var(--border);border-radius:16px;padding:16px 20px;margin-bottom:12px}
  .t1{font-size:1.7rem;font-weight:800}.t2{color:var(--muted)}
  .metric{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:12px}
  .metric h3{font-size:.85rem;color:#93a4b2;margin:0 0 6px}.metric .v{font-weight:800}
  .stTabs [data-baseweb="tab-list"]{gap:8px}
  .stTabs [data-baseweb="tab"]{padding:10px 16px;border-radius:10px;background:#0d1627;border:1px solid var(--border);color:var(--text)}
  .stTabs [aria-selected="true"]{background:#0f192b!important;border-color:#203049!important}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:12px}
  .pill{display:inline-block;padding:2px 8px;border:1px solid var(--border);background:#0c1423;border-radius:999px;font-size:12px;color:#cbd5e1}
  .calendar-card{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:8px}
  .stButton>button,.stDownloadButton>button{border-radius:12px}
  .side-hint{font-size:12px;color:#cbd5e1;background:#0d1627;border:1px solid #1e293b;border-radius:10px;padding:8px}
</style>
""", unsafe_allow_html=True)

# Estado
if "auth" not in st.session_state: st.session_state.auth=False
if "user" not in st.session_state: st.session_state.user=None
if "docs" not in st.session_state: st.session_state.docs: List[Dict[str,Any]]=[]
if "index" not in st.session_state: st.session_state.index=None
if "events" not in st.session_state: st.session_state.events=[]
if "log" not in st.session_state: st.session_state.log=[]
if "settings" not in st.session_state:
    st.session_state.settings={
        "project_name":"RAG-Agenda",
        "model":"openrouter/some-text-model",
        "vision_model":"openrouter/some-vision-model",
        "chunk_size":1000,
        "chunk_overlap":200,
        "top_k":5,
        "force_ocr_pdf":True,
        "read_method":"Auto",  # Auto | Solo OCR (Python) | Solo LLM visión
        "origen":"Cargados manualmente",
    }

def log_event(tipo, detalle, extra=None):
    st.session_state.log.append({"timestamp":datetime.now().isoformat(timespec="seconds"),
                                 "tipo":tipo,"detalle":detalle,"extra":extra or {}, "usuario":st.session_state.user})

def make_id():
    return base64.urlsafe_b64encode(os.urandom(9)).decode("utf-8").rstrip("=")

DATE_REGEXES=[r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"]
UUID_REGEX=r"\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\b"
RFC_REGEX=r"\b([A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3})\b"
TOTAL_REGEX=r"(?i)\b(total|importe|monto)\b\D{0,15}([\$\s]*[0-9]+[\,\.]?[0-9]{0,2})"
FOLIO_HINTS=r"(?i)(folio|factura|no\.?\s*\w*|id|referencia)\s*[:#\-]?\s*([A-Za-z0-9\-]{5,})"

def parse_dates(text):
    for rgx in DATE_REGEXES:
        m=re.search(rgx,text)
        if m:
            try: return dateparser.parse(m.group(1), dayfirst=True, fuzzy=True).date().isoformat()
            except Exception: pass
    return None

def extract_metadata(text, filename):
    md={"nombre":filename,"fecha":parse_dates(text) or None,"folio":None,"rfc":None,"total":None,"etiquetas":""}
    m=re.search(UUID_REGEX,text)
    if m: md["folio"]=m.group(0)
    else:
        m2=re.search(FOLIO_HINTS,text)
        if m2: md["folio"]=m2.group(2)[:40]
    m=re.search(RFC_REGEX,text)
    if m: md["rfc"]=m.group(1)
    m=re.search(TOTAL_REGEX,text)
    if m: md["total"]=m.group(2).strip()
    return md

def chunk_text(text, chunk_size, overlap):
    text=re.sub(r"\s+"," ",text).strip()
    if not text: return []
    out=[]; s=0; L=len(text)
    while s<L:
        e=min(L,s+chunk_size)
        sl=text[s:e]; p=max(sl.rfind(". "), sl.rfind("\n"))
        if p>100: e=s+p+1
        out.append(text[s:e])
        if e>=L: break
        s=max(0,e-overlap)
    return out

def build_index(docs, chunk_size, overlap):
    corpus=[]; mapping=[]
    for d in docs:
        for i,ch in enumerate(chunk_text(d.get("texto",""),chunk_size,overlap)):
            corpus.append(ch); mapping.append({"doc_id":d["id"],"chunk_idx":i})
    if not corpus: return None
    vec=TfidfVectorizer(ngram_range=(1,2), max_features=50000)
    mat=vec.fit_transform(corpus)
    st.session_state.index={"vectorizer":vec,"matrix":mat,"mapping":mapping}
    return st.session_state.index

def retrieve(query, top_k, origin_filter=None):
    idx=st.session_state.index
    if not idx: return []
    qv=idx["vectorizer"].transform([query])
    sims=cosine_similarity(qv, idx["matrix"]).ravel()
    order=np.argsort(-sims)
    out=[]
    for ti in order:
        m=idx["mapping"][ti]
        doc=next((d for d in st.session_state.docs if d["id"]==m["doc_id"]),None)
        if doc and (not origin_filter or doc.get("origen") in origin_filter):
            chs=chunk_text(doc.get("texto",""), st.session_state.settings["chunk_size"], st.session_state.settings["chunk_overlap"]) or [""]
            ctx=chs[m["chunk_idx"]] if m["chunk_idx"]<len(chs) else ""
            out.append({"score":float(sims[ti]),"doc":doc,"chunk":ctx})
            if len(out)>=top_k: break
    return out

# --- OCR / Visión
_OCR=None
def get_ocr():
    global _OCR
    if _OCR is None and easyocr is not None:
        try: _OCR=easyocr.Reader(["es","en"], gpu=False)
        except Exception: _OCR=None
    return _OCR

def preprocess(img: Image.Image) -> Image.Image:
    g=img.convert("L")
    w,h=g.size
    if max(w,h)<1600:
        scale=1600/max(w,h)
        g=g.resize((int(w*scale), int(h*scale)))
    g=ImageEnhance.Contrast(g).enhance(1.6)
    g=ImageEnhance.Sharpness(g).enhance(1.2)
    return g

def ocr_image_text(img: Image.Image) -> str:
    r=get_ocr()
    if r is None: return ""
    try:
        arr=np.array(preprocess(img))
        res=r.readtext(arr, detail=0, paragraph=True)
        return "\n".join([t.strip() for t in res if t and t.strip()])
    except Exception: return ""

def get_openrouter_key():
    return st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

def openrouter_chat(messages, model=None, temperature=0.1, max_tokens=800):
    key=get_openrouter_key()
    if not key: return ""
    model=model or st.session_state.settings["model"]
    url="https://openrouter.ai/api/v1/chat/completions"
    headers={"Authorization":f"Bearer {key}","Content-Type":"application/json",
             "HTTP-Referer":"https://streamlit.io","X-Title":"RAG-Docs-Agenda"}
    payload={"model":model,"messages":messages,"temperature":temperature,"max_tokens":max_tokens}
    try:
        r=requests.post(url,headers=headers,json=payload,timeout=60); r.raise_for_status()
        data=r.json(); return data.get("choices",[{}])[0].get("message",{}).get("content","").strip()
    except Exception: return ""

def vision_extract_from_image_bytes(b: bytes) -> str:
    key=get_openrouter_key(); vm=st.session_state.settings.get("vision_model")
    if not key or not vm: return ""
    b64=base64.b64encode(b).decode("utf-8")
    msg=[{"role":"user","content":[
        {"type":"text","text":"Extrae el texto visible del documento. Devuelve solo texto plano."},
        {"type":"input_image","image_data":b64}
    ]}]
    return openrouter_chat(msg, model=vm, max_tokens=2000)

def annotate_preview(img: Image.Image, text: str, max_chars=260):
    if img is None: return None
    W,H=img.size; box_h=int(H*0.22)
    overlay=img.copy(); draw=ImageDraw.Draw(overlay,'RGBA')
    draw.rectangle([(0,H-box_h),(W,H)], fill=(12,20,35,210))
    try: font=ImageFont.truetype("DejaVuSans.ttf", size=max(16,int(W*0.018)))
    except Exception: font=ImageFont.load_default()
    t=(text or "").replace("\r"," ").replace("\n"," ")
    t=t[:max_chars]+("…" if len(t)>max_chars else "")
    draw.multiline_text((18, H-box_h+18), t, fill=(226,232,240,255), font=font, spacing=4)
    return overlay

def extract_from_pdf_bytes(b: bytes) -> tuple[str, Image.Image|None, int]:
    if not fitz: return "", None, 0
    textos=[]; preview=None
    with fitz.open(stream=b, filetype="pdf") as doc:
        n=len(doc)
        for i,p in enumerate(doc):
            base_text=p.get_text("text") or ""
            need_ocr = st.session_state.settings["force_ocr_pdf"] or len(base_text.strip())<10 or st.session_state.settings["read_method"]!="Auto"
            if need_ocr:
                pix=p.get_pixmap(matrix=fitz.Matrix(2,2))
                img=Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
                if st.session_state.settings["read_method"]=="Solo LLM visión":
                    tb=vision_extract_from_image_bytes(pix.tobytes())  # may be large; acceptable for demo
                elif st.session_state.settings["read_method"]=="Solo OCR (Python)":
                    tb=ocr_image_text(img)
                else:
                    tb=ocr_image_text(img) or vision_extract_from_image_bytes(pix.tobytes()) or base_text
                base_text=tb or base_text
                if i==0: preview=img
            else:
                if i==0:
                    pix=p.get_pixmap(matrix=fitz.Matrix(2,2))
                    preview=Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
            textos.append(base_text)
        return "\n".join(textos), preview, n

def extract_from_image_bytes(b: bytes) -> tuple[str, Image.Image|None]:
    try: img=Image.open(io.BytesIO(b)).convert("RGB")
    except Exception: return "", None
    if st.session_state.settings["read_method"]=="Solo LLM visión":
        txt=vision_extract_from_image_bytes(b)
    elif st.session_state.settings["read_method"]=="Solo OCR (Python)":
        txt=ocr_image_text(img)
    else:
        txt=ocr_image_text(img) or vision_extract_from_image_bytes(b)
    return txt or "", img

# Login (Karla)
def login_box():
    st.sidebar.markdown("### 🔐 Acceso")
    with st.sidebar.form("login"):
        user=st.text_input("Usuario", value="Karla")
        pwd=st.text_input("Contraseña", type="password")
        ok=st.form_submit_button("Entrar")
    if ok:
        configured=st.secrets.get("APP_PASSWORD") or os.getenv("APP_PASSWORD") or "karla123"
        if user.strip().lower()=="karla" and pwd==configured:
            st.session_state.auth=True; st.session_state.user="Karla"; log_event("login","ok"); st.rerun()
        else:
            st.session_state.auth=False; st.error("Usuario o contraseña incorrectos.")

# Sidebar
def sidebar_config():
    st.sidebar.markdown("### ⚙️ Configuración")
    st.session_state.settings["project_name"]=st.sidebar.text_input("Proyecto", st.session_state.settings["project_name"])
    st.session_state.settings["model"]=st.sidebar.text_input("Modelo (OpenRouter, texto)", st.session_state.settings["model"])
    st.session_state.settings["vision_model"]=st.sidebar.text_input("Modelo visión (OpenRouter)", st.session_state.settings["vision_model"])
    st.session_state.settings["read_method"]=st.sidebar.selectbox("Método de lectura", ["Auto","Solo OCR (Python)","Solo LLM visión"], index=["Auto","Solo OCR (Python)","Solo LLM visión"].index(st.session_state.settings["read_method"]))
    st.session_state.settings["force_ocr_pdf"]=st.sidebar.toggle("Forzar OCR en PDFs", value=st.session_state.settings["force_ocr_pdf"])
    st.sidebar.divider()
    st.sidebar.markdown("**RAG**")
    st.session_state.settings["chunk_size"]=st.sidebar.slider("Tamaño de chunk",400,2000,st.session_state.settings["chunk_size"],step=50)
    st.session_state.settings["chunk_overlap"]=st.sidebar.slider("Solapamiento",0,400,st.session_state.settings["chunk_overlap"],step=20)
    st.session_state.settings["top_k"]=st.sidebar.slider("Top-K",1,10,st.session_state.settings["top_k"])
    st.sidebar.divider()
    st.session_state.settings["origen"]=st.sidebar.selectbox("Dependencia/Origen", ["Cargados manualmente","Dependencia A","Dependencia B","Correo","S3","Otro…"], index=0)
    with st.sidebar.expander("ℹ️ Ayuda rápida"):
        st.markdown("""
<div class="side-hint">
<b>Método de lectura</b>: elegir solo OCR (rápido local) o LLM visión (mejor en imágenes difíciles).<br>
<b>Chunk</b>: tamaño de corte del texto para recuperar contexto.<br>
<b>Solapamiento</b>: cuánto se repite entre chunks para no perder frases a caballo.<br>
<b>Top-K</b>: cuántos chunks relevantes usar para responder.
</div>
""", unsafe_allow_html=True)
    st.sidebar.divider()
    payload={"settings":st.session_state.settings,"docs":st.session_state.docs,"events":st.session_state.events,"log":st.session_state.log,"exported_at":datetime.now().isoformat(timespec="seconds")}
    st.download_button("💾 Exportar proyecto (.json)", json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name=f"{st.session_state.settings['project_name']}.json", mime="application/json")
    up=st.sidebar.file_uploader("Restaurar proyecto (.json)", type=["json"])
    if up is not None:
        try:
            data=json.load(io.TextIOWrapper(up, encoding="utf-8"))
            st.session_state.settings=data.get("settings", st.session_state.settings)
            st.session_state.docs=data.get("docs", [])
            st.session_state.events=data.get("events", [])
            st.session_state.log=data.get("log", [])
            st.success("Proyecto restaurado.")
        except Exception as e:
            st.error(f"Error al importar: {e}")

# Header
st.markdown('<div class="hdr"><div class="t1">🧠 RAG de Documentos · 🗓️ Agenda</div><div class="t2">Tema oscuro · Login Karla · OCR/Visión seleccionable · Texto general · RAG · Calendario</div></div>', unsafe_allow_html=True)
login_box()
if not st.session_state.auth: st.stop()
sidebar_config()

# Métricas
c1,c2,c3,c4=st.columns(4)
with c1: st.markdown(f'<div class="metric"><h3>Documentos</h3><div class="v">{len(st.session_state.docs)}</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric"><h3>Chunks</h3><div class="v">{(st.session_state.index["matrix"].shape[0] if st.session_state.index else 0)}</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric"><h3>Eventos</h3><div class="v">{len(st.session_state.events)}</div></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric"><h3>Bitácora</h3><div class="v">{len(st.session_state.log)}</div></div>', unsafe_allow_html=True)

# Tabs
TAB_DOCS, TAB_TEXT, TAB_RAG, TAB_CAL, TAB_LOG, TAB_ADMIN = st.tabs(
    ["📥 Documentos","🧾 Texto","🔎 RAG","🗓️ Calendario","📓 Bitácora","🧰 Admin"]
)

# Carga & extracción
with TAB_DOCS:
    st.subheader("Carga")
    ups=st.file_uploader("Arrastra PDFs/Imágenes (png, jpg, webp)", type=["pdf","png","jpg","jpeg","webp"], accept_multiple_files=True)
    if ups:
        for f in ups:
            doc_id=make_id()
            b=f.read()
            texto,preview,paginas="","",0
            if f.type=="application/pdf":
                t, prev, pags = extract_from_pdf_bytes(b)
                texto=t; preview=prev; paginas=pags
            else:
                t, img = extract_from_image_bytes(b)
                texto=t; preview=img; paginas=1
            md=extract_metadata(texto or "", f.name)
            doc={"id":doc_id,"nombre":md.get("nombre") or f.name,"origen":st.session_state.settings["origen"],
                 "texto":texto or "","paginas":paginas,"metadata":{"fecha":md.get("fecha"),"folio":md.get("folio"),"rfc":md.get("rfc"),"total":md.get("total"),"etiquetas":""},
                 "ts":datetime.now().isoformat(timespec="seconds")}
            st.session_state.docs.append(doc)
            log_event("doc:add", f.name, {"doc_id":doc_id,"origen":doc["origen"],"paginas":paginas})
            with st.expander(f"📄 {f.name}"):
                cprev,cmeta=st.columns([1,2])
                with cprev:
                    if isinstance(preview, Image.Image):
                        ann=annotate_preview(preview, texto)
                        st.image(ann or preview, caption=f"Vista previa anotada ({paginas or 1} pág)", use_container_width=True)
                        buf=io.BytesIO(); (ann or preview).save(buf, format="PNG")
                        st.download_button("⬇️ PNG anotado", buf.getvalue(), file_name=f"{os.path.splitext(f.name)[0]}_anotado.png", mime="image/png")
                    st.download_button("⬇️ Texto (.txt)", (texto or "").encode("utf-8"), file_name=f"{os.path.splitext(f.name)[0]}.txt", mime="text/plain")
                with cmeta:
                    editable=st.data_editor(pd.DataFrame([{
                        "id":doc_id,"Nombre":doc["nombre"],"Fecha":doc["metadata"]["fecha"],"Folio":doc["metadata"]["folio"],
                        "RFC":doc["metadata"]["rfc"],"Total":doc["metadata"]["total"],"Etiquetas":doc["metadata"]["etiquetas"],"Origen":doc["origen"]
                    }]), use_container_width=True, hide_index=True)
                    row=editable.iloc[0]
                    doc["nombre"]=row["Nombre"]; doc["origen"]=row["Origen"]
                    doc["metadata"].update({"fecha":row["Fecha"],"folio":row["Folio"],"rfc":row["RFC"],"total":row["Total"],"etiquetas":row["Etiquetas"]})
                    st.text_area("Texto extraído (resumen)", (texto or "")[:2500], height=200)

    st.divider()
    st.subheader("Tabla")
    if st.session_state.docs:
        df=pd.DataFrame([{ "ID":d["id"],"Nombre":d["nombre"],"Origen":d["origen"],"Fecha":d["metadata"].get("fecha"),
                           "Folio":d["metadata"].get("folio"),"RFC":d["metadata"].get("rfc"),"Total":d["metadata"].get("total"),
                           "Páginas":d.get("paginas",0),"Cargado":d.get("ts"),"Etiquetas":d["metadata"].get("etiquetas","")} for d in st.session_state.docs])
        edited=st.data_editor(df, hide_index=True, use_container_width=True)
        for _,row in edited.iterrows():
            d=next((x for x in st.session_state.docs if x["id"]==row["ID"]),None)
            if d:
                d["nombre"]=row["Nombre"]; d["origen"]=row["Origen"]
                d["metadata"].update({"fecha":row["Fecha"],"folio":row["Folio"],"rfc":row["RFC"],"total":row["Total"],"etiquetas":row.get("Etiquetas","")})
        st.download_button("⬇️ Exportar CSV", edited.to_csv(index=False).encode("utf-8"), file_name="documentos.csv", mime="text/csv")
    else:
        st.info("Carga documentos para comenzar.")

# TEXTO general
with TAB_TEXT:
    st.subheader("Texto general")
    if not st.session_state.docs:
        st.info("Sin documentos.")
    else:
        options=[f"{d['nombre']} — {d['id'][:6]}" for d in st.session_state.docs]
        sel=st.selectbox("Documento", options)
        dsel=next(d for d in st.session_state.docs if sel.startswith(d["nombre"]))
        colA,colB=st.columns([3,1])
        with colA:
            txt_area=st.text_area("Contenido completo", dsel.get("texto",""), height=520, key=f"txt_{dsel['id']}")
        with colB:
            st.write("**Metadatos**"); st.json({"Nombre":dsel["nombre"], **dsel["metadata"], "Páginas":dsel.get("paginas",0)}, expanded=False)
            st.download_button("⬇️ Descargar .txt", (dsel.get("texto","")).encode("utf-8"), file_name=f"{os.path.splitext(dsel['nombre'])[0]}.txt")
            st.write("**Buscar**")
            q=st.text_input("Palabra/frase")
            if q:
                txt=dsel.get("texto",""); matches=[m.start() for m in re.finditer(re.escape(q), txt, flags=re.IGNORECASE)]
                st.caption(f"Coincidencias: {len(matches)}")
                for i,pos in enumerate(matches[:10],1):
                    s=max(0,pos-80); e=min(len(txt),pos+80); st.markdown(f"**{i}.** …{txt[s:e]}…")
        if st.button("Re-extraer con método actual"):
            # re-extracción rápida si hay preview/bytes (no guardamos bytes; pedimos de nuevo al usuario si es necesario)
            st.info("Carga nuevamente el archivo para re-extraer con el método seleccionado en la barra lateral.")

# RAG
with TAB_RAG:
    st.subheader("Buscador + QA")
    cA,cB=st.columns([1,2])
    with cA:
        if st.button("🧱 (Re)construir índice", type="primary"):
            idx=build_index(st.session_state.docs, st.session_state.settings["chunk_size"], st.session_state.settings["chunk_overlap"])
            if idx: st.success(f"Índice listo ({idx['matrix'].shape[0]} chunks).")
            else: st.warning("No hay texto indexable.")
        filtros=sorted(set(d.get("origen") for d in st.session_state.docs))
        f_origen=st.multiselect("Filtrar por origen", filtros)
    with cB:
        q=st.text_input("Pregunta", placeholder="¿Cuál es el total de la factura X?")
        if q:
            ctxs=retrieve(q, st.session_state.settings["top_k"], origin_filter=f_origen if f_origen else None) if st.session_state.index else []
            if not ctxs: st.warning("Construye el índice o carga documentos.")
            else:
                blob="\n\n".join([f"[Doc: {c['doc']['nombre']} | Fecha: {c['doc']['metadata'].get('fecha')} | Folio: {c['doc']['metadata'].get('folio')}]\n{c['chunk']}" for c in ctxs])
                msgs=[{"role":"system","content":"Responde solo con el contexto dado. Cita nombre/fecha/folio si aparece."},
                      {"role":"user","content":f"Contexto:\n{blob}\n\nPregunta: {q}"}]
                ans=openrouter_chat(msgs, model=st.session_state.settings["model"], temperature=0.1, max_tokens=700)
                st.markdown("### Respuesta"); st.write(ans or "(sin respuesta)")
                with st.expander("Contexto usado"):
                    for i,c in enumerate(ctxs,1):
                        meta=c['doc']['metadata']; st.markdown(f"**{i}. {c['doc']['nombre']}** · score={c['score']:.3f} · Fecha: {meta.get('fecha')} · Folio: {meta.get('folio')}")
                        st.write(c["chunk"][:1600])

# Calendario
with TAB_CAL:
    st.subheader("Agenda")
    colL,colR=st.columns([2,1])
    with colL:
        if calendar is None:
            st.info("Instala 'streamlit-calendar' para el calendario interactivo.")
        else:
            options={"editable":True,"selectable":True,"initialView":"dayGridMonth","locale":"es","height":760,
                     "nowIndicator":True,"dayMaxEventRows":3,
                     "eventTimeFormat":{"hour":"2-digit","minute":"2-digit","meridiem":False},
                     "headerToolbar":{"left":"prev,next today","center":"title","right":"dayGridMonth,timeGridWeek,timeGridDay,listWeek"}}
            st.markdown('<div class="calendar-card">', unsafe_allow_html=True)
            cal=calendar(events=st.session_state.events or [], options=options, key=f"cal_{len(st.session_state.events)}")
            st.markdown('</div>', unsafe_allow_html=True)

            if cal:
                # clicks
                if cal.get("eventClick"):
                    ev=cal["eventClick"]["event"]; st.toast(f"{ev['title']} — {ev['start']}")
                # selección de rango (algunas versiones exponen 'select' o 'selected')
                sel=cal.get("select") or cal.get("selected")
                if sel and st.session_state.get("allow_add_from_select", True):
                    try:
                        s=sel["start"]; e=sel.get("end", s)
                        ev={"title":"Nuevo evento","start":s,"end":e,"color":"#60a5fa","extendedProps":{"creado_por":st.session_state.user}}
                        st.session_state.events.append(ev); log_event("calendar:add","Nuevo (UI)")
                        st.session_state.allow_add_from_select=False
                        st.rerun()
                    except Exception: pass
                else:
                    st.session_state.allow_add_from_select=True
    with colR:
        with st.form("add_event"):
            titulo=st.text_input("Título","Revisión de documentos")
            inicio=st.date_input("Inicio", value=date.today())
            hora_ini=st.time_input("Hora inicio", value=datetime.now().time().replace(second=0, microsecond=0))
            fin=st.date_input("Fin", value=date.today())
            hora_fin=st.time_input("Hora fin", value=(datetime.now()+timedelta(hours=1)).time().replace(second=0, microsecond=0))
            color=st.selectbox("Color", ["#60a5fa","#34d399","#f59e0b","#ef4444","#8b5cf6"])
            notas=st.text_area("Notas")
            if st.form_submit_button("➕ Añadir"):
                ev={"title":titulo,"start":datetime.combine(inicio,hora_ini).isoformat(),
                    "end":datetime.combine(fin,hora_fin).isoformat(),"color":color,
                    "extendedProps":{"notas":notas,"creado_por":st.session_state.user}}
                st.session_state.events.append(ev); log_event("calendar:add", titulo); st.success("Evento agregado."); st.rerun()
        if st.session_state.events:
            st.markdown("#### Próximos 7 días")
            def parse_iso(ts):
                try: return datetime.fromisoformat(ts)
                except Exception: return None
            upcoming=[(parse_iso(e["start"]), e) for e in st.session_state.events if e.get("start")]
            upcoming=[x for x in upcoming if x[0] is not None]; upcoming=sorted(upcoming,key=lambda x:x[0])
            now=datetime.now()
            future=[e for t,e in upcoming if 0 <= (t-now).days <= 7][:8]
            for e in future:
                st.markdown(f"<div class='card'><b>{e['title']}</b><br><span class='pill'>{e['start']}</span></div>", unsafe_allow_html=True)
            st.download_button("📥 Exportar eventos (JSON)", json.dumps(st.session_state.events, ensure_ascii=False, indent=2), "eventos.json")

# Bitácora
with TAB_LOG:
    st.subheader("Bitácora")
    if st.session_state.log:
        df=pd.DataFrame(st.session_state.log)
        tipo=st.selectbox("Filtrar", options=["(todos)"]+sorted(df.tipo.unique().tolist()))
        show=df if tipo=="(todos)" else df[df.tipo==tipo]
        st.dataframe(show.sort_values("timestamp", ascending=False), use_container_width=True)
        st.download_button("⬇️ Exportar", show.to_csv(index=False).encode("utf-8"), "bitacora.csv")
    else:
        st.info("Aún no hay eventos.")

# Admin
with TAB_ADMIN:
    st.subheader("Mantenimiento")
    c1,c2,c3=st.columns(3)
    if c1.button("🧹 Limpiar documentos"): st.session_state.docs=[]; st.session_state.index=None; log_event("admin:clear_docs",""); st.success("Documentos eliminados.")
    if c2.button("🧹 Limpiar bitácora"): st.session_state.log=[]; st.success("Bitácora limpia.")
    if c3.button("🧹 Limpiar eventos"): st.session_state.events=[]; st.success("Eventos eliminados.")
