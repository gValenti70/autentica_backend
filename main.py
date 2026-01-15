import os
import json
import time
import re
from datetime import datetime, timezone
from typing import Optional, Tuple, Any, Dict, List, Literal
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from pydantic import BaseModel
from difflib import SequenceMatcher
from openai import AzureOpenAI
import logging
from PIL import Image
import pillow_avif  # noqa: F401
import base64
import io
import socket
import uvicorn
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from pymongo import ReturnDocument
from bson import ObjectId
import bcrypt
import hashlib

# ======================================================
# LOGGING
# ======================================================
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ======================================================
# CONFIG OPENAI
# ======================================================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://autenticagpt.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
DEPLOYMENT_GPT = os.getenv("DEPLOYMENT_GPT", "gpt-5.1-chat")

client_gpt = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

MAX_FOTO = int(os.getenv("MAX_FOTO", "7"))

# ======================================================
# MONGO CONFIG
# ======================================================
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "autentica")


# Collezioni (nomenclatura chiara)
analisi_col = "aut_analisi"
foto_col = "aut_analisi_foto"
prompts_col = "aut_prompts"
prompt_versions_col = "aut_prompt_versions"
users_col = "aut_users"
vademecum_col = "aut_vademecum"
login_log_col = "aut_login_log"


_mongo_client: Optional[MongoClient] = None

def get_mongo_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        if not MONGO_URI:
            raise RuntimeError("MONGO_URI missing (set it in App Service env vars)")
        # DocumentDB Mongo compat: TLS obbligatorio, SRV ok
        _mongo_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=8000,
            connectTimeoutMS=8000,
            socketTimeoutMS=20000,
            retryWrites=False
        )
    return _mongo_client

def get_db():
    client = get_mongo_client()
    return client[MONGO_DB_NAME]

def safe_iso_datetime(value):
    if value is None:
        return None

    # datetime Mongo
    if isinstance(value, datetime):
        return value.isoformat()

    # epoch float / int
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        except Exception:
            return None

    # stringa già formattata
    if isinstance(value, str):
        return value

    return None


# ======================================================
# FASTAPI
# ======================================================
app = FastAPI(title="Autentica V2 Backend", version="6.0-mongo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.on_event("startup")
def startup():
    try:
        client = get_mongo_client()
        client.admin.command("ping")
        logger.info("[MONGO] Connessione OK")

        # ⚠️ COMMENTARE in ambienti senza permessi admin
        # ensure_indexes()

    except Exception as e:
        logger.error(f"[MONGO] Startup error: {e}")


# ======================================================
# INPUT MODEL
# ======================================================
class InputAnalisi(BaseModel):
    tipologia: Optional[str] = "borsa"
    modello: Optional[str] = "generico"
    foto: str
    id_analisi: Optional[str]
    user_id: Optional[str] = "default"

# ======================================================
# UTILS
# ======================================================
# def similarity(a, b):
#     return SequenceMatcher(None, a, b).ratio()

def normalize(t):
    if not t:
        return ""
    t = t.lower()
    for sep in [" ", "-", "_"]:
        t = t.replace(sep, "")
    return t

def json_safe(obj: Any) -> Any:
    """Converte ricorsivamente tipi non JSON-serializzabili (ObjectId, datetime, ecc.)."""
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return safe_iso_datetime(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(x) for x in obj]
    return obj


def to_dt_utc(x: Any) -> str:
    """Ritorna una ISO string robusta anche se in DB hai float epoch."""
    if isinstance(x, datetime):
        return safe_iso_datetime(x)
    if isinstance(x, (int, float)):
        return safe_iso_datetime(datetime.fromtimestamp(x, tz=timezone.utc))
    return safe_iso_datetime(datetime.now(timezone.utc))


# ======================================================
# AVIF -> JPEG
# ======================================================
def convert_avif_to_jpeg(base64_data: str) -> str:
    image_bytes = base64.b64decode(base64_data)
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    jpeg_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return jpeg_base64



# ======================================================
# PROMPT SYSTEM (Mongo)
# ======================================================
def load_prompt_from_db(alias: str, user_id: str):
    db = get_db()

    # 1) prova user-specific
    prompt = db[prompt_versions_col].find_one(
        {"prompt_name": alias, "user_id": user_id, "is_active": True},
        sort=[("version", -1), ("created_at", -1)]
    )

    # 2) fallback globale (user_id assente o "default")
    if not prompt:
        prompt = db[prompt_versions_col].find_one(
            {"prompt_name": alias, "is_active": True},
            sort=[("version", -1), ("created_at", -1)]
        )

    if not prompt:
        raise ValueError(f"Prompt alias '{alias}' non trovato in {prompt_versions_col} (is_active=true)")

    content = prompt.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Prompt '{alias}' trovato ma 'content' è vuoto/invalid")

    return content, prompt

def load_guardrail(user_id: str):
    try:
        return load_prompt_from_db("vision_guardrail", user_id)
    except:
        return "Non parlare mai di persone o privacy.", {}


def tokenize_model_name(name: str) -> list[str]:
    if not name:
        return []
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    return name.split()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()



vlog = logging.getLogger("VADEMECUM")
vlog.setLevel(logging.INFO)

def load_vademecum_mongo(model: str, brand: str, db):
    meta = {
        "model_requested": model or "",
        "model_norm": normalize(model or ""),
        "brand": brand,
        "file": None,
        "path": None,
        "source": None,
        "match_type": None,
        "fuzzy_score": None,
        "length_chars": None,
        "debug": {}
    }

    brand_raw = brand or ""
    model_raw = model or ""

    brand_norm = normalize(brand_raw)
    model_norm = normalize(model_raw)

    col = db["aut_vademecum"]

    # ---- LOG INPUT
    vlog.info("========== VADEMECUM LOOKUP ==========")
    vlog.info(f"[IN ] brand_raw='{brand_raw}' model_raw='{model_raw}'")
    vlog.info(f"[NORM] brand_norm='{brand_norm}' model_norm='{model_norm}'")

    # ---- sanity: ping DB + collection stats (light)
    try:
        # count_documents su filtro piccolo (brand) per capire se stai nel DB giusto
        brand_count = col.count_documents({"brand_norm": brand_norm}) if brand_norm else None
        total_models = col.count_documents({"type": "model"})
        vlog.info(f"[DB ] total(type=model)={total_models} | brand_count={brand_count}")
        meta["debug"]["total_models"] = total_models
        meta["debug"]["brand_count"] = brand_count
    except Exception as e:
        vlog.error(f"[DB ] count_documents error: {e}")
        meta["debug"]["count_error"] = str(e)

    # ---- tokenizzazione (per debug e fallback)
    tokens = re.sub(r"[^a-zA-Z0-9\s]", " ", model_raw.lower()).split()
    tokens_norm = [normalize(t) for t in tokens if len(t) >= 2]

    vlog.info(f"[TOK] tokens={tokens}")
    vlog.info(f"[TOK] tokens_norm={tokens_norm}")

    meta["debug"]["tokens"] = tokens
    meta["debug"]["tokens_norm"] = tokens_norm

    # ======================================================
    # 1) Exact model (brand_norm + model_norm)
    # ======================================================
    if brand_norm and model_norm:
        q1 = {"brand_norm": brand_norm, "model_norm": model_norm, "type": "model"}
        vlog.info(f"[Q1 ] exact query={q1}")

        try:
            doc = col.find_one(q1, {"raw_text": 1, "brand_norm": 1, "model_norm": 1, "model": 1, "type": 1})
        except Exception as e:
            vlog.error(f"[Q1 ] find_one error: {e}")
            doc = None
            meta["debug"]["q1_error"] = str(e)

        if doc:
            vlog.info(f"[Q1 ] HIT doc.model_norm='{doc.get('model_norm')}' doc.model='{doc.get('model')}'")
        else:
            vlog.info("[Q1 ] MISS")

        if doc and doc.get("raw_text"):
            text = doc["raw_text"]
            meta.update({
                "source": "model_exact",
                "match_type": "model_exact",
                "vademecum_id": str(doc.get("_id")),
                "length_chars": len(text)
            })
            meta["debug"]["q1_hit"] = True
            return text, meta

    # ======================================================
    # 1b) Exact token (brand_norm + token_norm)
    # ======================================================
    if brand_norm and tokens_norm:
        vlog.info("[Q1b] trying exact token matches...")
        for t in tokens_norm[:10]:  # limit log noise
            q1b = {"brand_norm": brand_norm, "model_norm": t, "type": "model"}
            try:
                doc = col.find_one(q1b, {"raw_text": 1, "model_norm": 1, "model": 1})
            except Exception as e:
                vlog.error(f"[Q1b] find_one error on token='{t}': {e}")
                continue

            vlog.info(f"[Q1b] token='{t}' -> {'HIT' if doc else 'MISS'}")
            if doc and doc.get("raw_text"):
                text = doc["raw_text"]
                meta.update({
                    "source": "model_exact_token",
                    "match_type": "model_exact",
                    "vademecum_id": str(best.get("_id")),
                    "length_chars": len(text)
                })
                meta["debug"]["q1b_token_hit"] = t
                return text, meta

    # ======================================================
    # 2) Fuzzy model (within brand)
    # ======================================================
    if brand_norm and (model_norm or tokens_norm):
        q2 = {"brand_norm": brand_norm, "type": "model"}
        vlog.info(f"[Q2 ] loading candidates query={q2}")

        try:
            candidates = list(col.find(q2, {"model": 1, "model_norm": 1, "raw_text": 1}))
            vlog.info(f"[Q2 ] candidates_count={len(candidates)}")
            meta["debug"]["candidates_count"] = len(candidates)
        except Exception as e:
            vlog.error(f"[Q2 ] find error: {e}")
            candidates = []
            meta["debug"]["q2_error"] = str(e)

        best = None
        best_score = 0.0
        best_token = None

        # confronto sia su model_norm intero sia token
        probes = [model_norm] if model_norm else []
        probes += tokens_norm[:10]

        for c in candidates:
            c_norm = c.get("model_norm", "") or ""
            for p in probes:
                score = similarity(p, c_norm)
                if score > best_score:
                    best = c
                    best_score = score
                    best_token = p

        vlog.info(f"[Q2 ] best_score={best_score:.3f} best_token='{best_token}' best_candidate='{(best or {}).get('model_norm')}'")

        # if best and best_score >= 0.60 and best.get("raw_text"):
        #     text = best["raw_text"]
        #     meta.update({
        #         "source": "model_fuzzy",
        #         "match_type": "model_fuzzy",
        #         "vademecum_id": str(doc.get("_id")),
        #         "fuzzy_score": round(best_score, 3),
        #         "length_chars": len(text)
        #     })
        #     meta["debug"]["q2_best_token"] = best_token
        #     meta["debug"]["q2_best_model_norm"] = best.get("model_norm")
        #     return text, meta
        if best and best_score >= 0.60 and best.get("raw_text"):
            text = best["raw_text"]
            meta.update({
                "source": "model_fuzzy",
                "match_type": "model_fuzzy",
                "vademecum_id": str(doc.get("_id")),
                "fuzzy_score": round(best_score, 3),
                "length_chars": len(text)
            })
            meta["debug"]["q2_best_token"] = best_token
            meta["debug"]["q2_best_model_norm"] = best.get("model_norm")
            return text, meta

    # ======================================================
    # 3) Brand generic
    # ======================================================
    if brand_norm:
        q3 = {"brand_norm": brand_norm, "type": "brand_generic"}
        vlog.info(f"[Q3 ] brand_generic query={q3}")
        try:
            doc = col.find_one(q3, {"raw_text": 1})
        except Exception as e:
            vlog.error(f"[Q3 ] find_one error: {e}")
            doc = None
            meta["debug"]["q3_error"] = str(e)

        if doc and doc.get("raw_text"):
            text = doc["raw_text"]
            meta.update({
                "source": "brand_generic",
                "match_type": "brand_generic",
                "vademecum_id": str(doc.get("_id")),
                "length_chars": len(text)
            })
            meta["debug"]["q3_hit"] = True
            return text, meta

    # ======================================================
    # 4) General fallback
    # ======================================================
    q4 = {"type": "general"}
    vlog.info(f"[Q4 ] general query={q4}")
    try:
        doc = col.find_one(q4, {"raw_text": 1})
    except Exception as e:
        vlog.error(f"[Q4 ] find_one error: {e}")
        doc = None
        meta["debug"]["q4_error"] = str(e)

    if doc and doc.get("raw_text"):
        text = doc["raw_text"]
        meta.update({
            "source": "fallback_general",
            "match_type": "general",
            "length_chars": len(text)
        })
        meta["debug"]["q4_hit"] = True
        return text, meta

    vlog.info("[OUT] fallback_hardcoded")
    meta.update({"source": "fallback_hardcoded", "match_type": "hardcoded", "length_chars": None})
    return """Verificare con attenzione logo, cuciture, hardware, materiali, simmetria ed eventuali codici o seriali, 
            ricercando attivamente discrepanze rispetto agli standard noti del brand. 
            Anche imprecisioni lievi, incoerenze di allineamento, variazioni di qualità o finitura, 
            o differenze nella resa dei dettagli devono essere considerate come potenziali segnali di non originalità 
            e incidere sulla valutazione complessiva.
    """, meta



# ======================================================
# PROMPTSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
# ======================================================

def prompt_get_active(prompt_name: str, user_id: str = "default"):
    db = get_db()

    doc = db.aut_prompt_versions.find_one(
        {
            "prompt_name": prompt_name,
            "user_id": user_id,
            "is_active": True
        },
        sort=[("version", -1)]
    )

    if not doc:
        return None

    return {
        "prompt_name": doc["prompt_name"],
        "content": doc["content"],
        "version": doc["version"],
        "user_id": doc["user_id"],
        "is_active": doc["is_active"],
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at")
    }
    

def prompt_save_new_version(
    prompt_name: str,
    content: str,
    user_id: str = "default",
    comment: str = "",
    feedback: str = ""
    
):
    if not content or not content.strip():
        raise ValueError("Prompt vuoto")

    db = get_db()

    # disattiva versioni precedenti
    db.aut_prompt_versions.update_many(
        {
            "prompt_name": prompt_name,
            "user_id": user_id
        },
        {"$set": {"is_active": False}}
    )

    last = db.aut_prompt_versions.find_one(
        {
            "prompt_name": prompt_name,
            "user_id": user_id
        },
        sort=[("version", -1)]
    )

    new_version = (last["version"] if last else 0) + 1
    now = datetime.now(timezone.utc)

    db.aut_prompt_versions.insert_one({
        "prompt_name": prompt_name,
        "content": content,
        "version": new_version,
        "user_id": user_id,
        "comment": comment,
        "feedback": feedback,
        "is_active": True,
        "created_at": now,
        "updated_at": now
    })

    return {
        "prompt_name": prompt_name,
        "new_version": new_version
    }

def prompt_list_active(user_id: str):
    db = get_db()

    pipeline = [
        {"$match": {
            "is_active": True,
            "user_id": user_id        # 🔴 FILTRO FONDAMENTALE
        }},
        {"$group": {
            "_id": "$prompt_name",
            "active_version": {"$max": "$version"}
        }},
        {"$sort": {"_id": 1}}
    ]

    rows = list(db.aut_prompt_versions.aggregate(pipeline))

    return [
        {
            "prompt_name": r["_id"],
            "active_version": r["active_version"],
            "user_id": user_id
        }
        for r in rows
    ]
    
def prompt_history(prompt_name: str, user_id: str):
    db = get_db()

    rows = list(db.aut_prompt_versions.find(
        {
            "prompt_name": prompt_name,
            "user_id": user_id
        },
        {
            "_id": 1,
            "version": 1,
            "content": 1,
            "comment": 1,
            "feedback": 1,
            "is_active": 1,
            "created_at": 1
        }
    ).sort("version", -1))

    return [
        {
            "id": str(r["_id"]),
            "version": r.get("version"),
            "content": r.get("content"),
            "comment": r.get("comment"),
            "feedback": r.get("feedback"),
            "is_active": r.get("is_active"),
            "created_at": r.get("created_at")
        }
        for r in rows
    ]


class PromptActivateInput(BaseModel):
    user_id: str = "default"
    prompt_name: str
    version: int

class PromptDeleteInput(BaseModel):
    user_id: str = "default"
    id: str

class PromptSaveInput(BaseModel):
    prompt_name: str
    content: str
    user_id: str = "default"
    comment: Optional[str] = ""
    feedback: Optional[str] = ""

class PromptFeedbackInput(BaseModel):
    user_id: str = "default"
    id: str
    feedback: str

# ======================================================
# ENDPOINTS PROMPTSSSSSSSSSSSSSSSS
# ======================================================

@app.post("/prompt/feedback")
def api_prompt_update_feedback(data: PromptFeedbackInput):
    db = get_db()

    try:
        oid = ObjectId(data.id)
    except Exception:
        raise HTTPException(status_code=400, detail="ID non valido")

    res = db.aut_prompt_versions.update_one(
        {
            "_id": oid,
            "user_id": data.user_id
        },
        {
            "$set": {
                "feedback": data.feedback,
                "updated_at": datetime.now(timezone.utc)
            }
        }
    )

    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Versione non trovata")

    return {
        "status": "ok",
        "id": data.id,
        "feedback": data.feedback
    }


@app.post("/prompt/delete")
def api_prompt_delete(data: PromptDeleteInput):
    db = get_db()

    try:
        oid = ObjectId(data.id)
    except Exception:
        raise HTTPException(status_code=400, detail="ID non valido")

    # recupera documento
    doc = db.aut_prompt_versions.find_one({
        "_id": oid,
        "user_id": data.user_id
    })

    if not doc:
        raise HTTPException(status_code=404, detail="Versione non trovata")

    if doc.get("is_active") is True:
        raise HTTPException(
            status_code=400,
            detail="Impossibile eliminare la versione attiva"
        )

    db.aut_prompt_versions.delete_one({"_id": oid})

    return {
        "status": "ok",
        "deleted_id": data.id,
        "prompt_name": doc.get("prompt_name"),
        "version": doc.get("version")
    }


@app.post("/prompt/activate")
def api_prompt_activate(data: PromptActivateInput):
    db = get_db()

    # disattiva tutte le versioni
    db.aut_prompt_versions.update_many(
        {
            "user_id": data.user_id,
            "prompt_name": data.prompt_name
        },
        {
            "$set": {
                "is_active": False,
                "updated_at": datetime.now(timezone.utc)
            }
        }
    )

    # attiva la versione richiesta
    res = db.aut_prompt_versions.update_one(
        {
            "user_id": data.user_id,
            "prompt_name": data.prompt_name,
            "version": data.version
        },
        {
            "$set": {
                "is_active": True,
                "updated_at": datetime.now(timezone.utc)
            }
        }
    )

    if res.matched_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Versione non trovata per questo user/prompt/version"
        )

    return {
        "status": "ok",
        "prompt_name": data.prompt_name,
        "activated_version": data.version
    }

@app.post("/prompt/save")
def api_prompt_save(data: PromptSaveInput):
    try:
        res = prompt_save_new_version(
            data.prompt_name,
            data.content,
            data.user_id,
            data.comment,
            data.feedback
        )
        return {"status": "ok", **res}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/prompt/get/{prompt_name}")
def api_prompt_get(prompt_name: str, user_id: str = "default"):
    res = prompt_get_active(prompt_name, user_id)
    if not res:
        raise HTTPException(404, "Prompt non trovato")
    return json_safe(res)
    
@app.get("/prompt/list")
def api_prompt_list(user_id: str = "default"):
    return prompt_list_active(user_id)

@app.get("/prompt/history/{prompt_name}")
def api_prompt_history(prompt_name: str, user_id: str = "default"):
    res = prompt_history(prompt_name, user_id)
    if not res:
        raise HTTPException(404, "Nessuna versione trovata")
    return res

@app.get("/prompt/list_active")
def api_prompt_list_active(exclude_user_id: str | None = None, limit: int = 500):
    db = get_db()

    q = {"is_active": True}
    if exclude_user_id:
        q["user_id"] = {"$ne": exclude_user_id}

    cur = (
        db.aut_prompt_versions
        .find(
            q,
            {
                "_id": 0,
                "prompt_name": 1,
                "user_id": 1,
                "version": 1,
                "created_at": 1,
                "updated_at": 1
            }
        )
        # 👇 ORDINAMENTO DECENTE
        .sort([
            ("user_id", 1),        # A → Z utenti
            ("prompt_name", 1)     # A → Z prompt
        ])
        .limit(int(limit))
    )

    return list(cur)
# ======================================================
# ENDPOINTS BACKEND
# ======================================================
@app.post("/analizza-oggetto")
async def analizza_oggetto(input: InputAnalisi):
    db = get_db()

    # ✅ tipologia: non forzare più "borsa" qui
    tipologia_input = (input.tipologia or "").strip()

    user_id = (input.user_id or "default").strip()

    # =========================
    # 0) Validazione / recupero analisi
    # =========================
    oid = None
    analisi = None

    if input.id_analisi:
        try:
            oid = ObjectId(input.id_analisi)
        except Exception:
            raise HTTPException(status_code=422, detail="id_analisi non valido (ObjectId richiesto)")

        analisi = db[analisi_col].find_one({"_id": oid})
        if not analisi:
            raise HTTPException(status_code=404, detail="Analisi non trovata")

    if oid is None:
        res = db[analisi_col].insert_one({
            "user_id": user_id,
            "stato": "in_corso",
            "step_corrente": 0,
            # ✅ aggiunto: tipologia salvata subito (fallback iniziale)
            "tipologia": tipologia_input or "borsa",
            "marca_stimata": None,
            "modello_stimato": None,
            "percentuale_contraffazione": None,
            "giudizio_finale": None,
            "created_at": datetime.now(timezone.utc)
        })
        oid = res.inserted_id
        analisi = db[analisi_col].find_one({"_id": oid})

    # =========================
    # 1) Salva foto (base64 + AVIF->JPG)
    # =========================
    base64_pura = (input.foto or "").split(",")[-1].strip()
    if not base64_pura:
        raise HTTPException(status_code=422, detail="foto base64 mancante")

    try:
        raw = base64.b64decode(base64_pura)
        is_avif = (raw[4:8] == b'ftyp' and b'avif' in raw[:32])
    except Exception:
        is_avif = False

    foto_b64 = convert_avif_to_jpeg(base64_pura) if is_avif else base64_pura

    step_corrente = db[foto_col].count_documents({"id_analisi": oid}) + 1

    db[foto_col].insert_one({
        "id_analisi": oid,
        "step": step_corrente,
        "foto_base64": foto_b64,
        "json_response": None,
        "created_at": datetime.now(timezone.utc)
    })

    immagini = [
        r["foto_base64"]
        for r in db[foto_col]
            .find({"id_analisi": oid}, {"_id": 0, "foto_base64": 1, "step": 1})
            .sort("step", 1)
    ]
    num_foto = len(immagini)

    # =========================
    # 2) Marca/modello dal DB
    # =========================
    marca = (analisi or {}).get("marca_stimata")
    modello = (analisi or {}).get("modello_stimato")

    # =========================
    # ✅ 2bis) Tipologia dal DB (se già determinata)
    # =========================
    tipologia_db = (analisi or {}).get("tipologia")
    tipologia = (tipologia_db or tipologia_input or "borsa").strip()

    # =========================
    # 3) Vademecum
    # =========================
    # vademecum_text, vmeta = load_vademecum(modello or "", marca)
    vademecum_text, vmeta = load_vademecum_mongo(modello or "", marca, db)

    # =========================
    # 4) Scelta prompt (finale SOLO se GPT ha detto basta)
    # =========================
    if step_corrente == 1:
        prompt_name = "step1_identificazione"

    elif step_corrente >= MAX_FOTO:
        # 🔒 SETTIMA FOTO → FORZA PROMPT FINALE
        prompt_name = "step3_finale"

    else:
        prev = db[foto_col].find_one(
            {"id_analisi": oid, "step": step_corrente - 1},
            {"_id": 0, "json_response": 1}
        )
        prev_json = prev.get("json_response") if isinstance(prev, dict) else None

        if isinstance(prev_json, dict) and prev_json.get("richiedi_altra_foto") is False:
            prompt_name = "step3_finale"
        else:
            prompt_name = "step2_intermedio"

    base_prompt, meta_prompt = load_prompt_from_db(prompt_name, user_id)
    guardrail, _ = load_guardrail(user_id)

    modello_safe = str(modello) if modello else ""

    final_prompt = (
        base_prompt
        .replace("{{GUARDRAIL}}", guardrail)
        .replace("{{TIPOLOGIA}}", tipologia)  # ✅ ora è quella “vera” (db > input > fallback)
        .replace("{{MODELLO}}", modello_safe)
        .replace("{{NUM_FOTO}}", str(num_foto))
        .replace("{{VADEMECUM}}", vademecum_text)
        .replace("{{JSON_RULE}}", "Rispondi SOLO con JSON valido.")
    )

    # =========================
    # 5) GPT CALL
    # =========================
    t0 = time.time()
    messages = [{
        "role": "user",
        "content": (
            [{"type": "text", "text": final_prompt}] +
            [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in immagini]
        )
    }]

    resp = client_gpt.chat.completions.create(
        model=DEPLOYMENT_GPT,
        messages=messages
    )
    t1 = time.time()

    timing = {
        "tempo_chat_gpt_ms": round((t1 - t0) * 1000, 2)
    }

    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1].replace("json", "").strip()

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("JSON non è un oggetto")
    except Exception:
        data = {
            "percentuale": 100,
            "motivazione": "Risposta AI non valida",
            "richiedi_altra_foto": False,
            "errore_raw": raw
        }

    # =========================
    # 6) Blocca marca/modello/tipologia dopo step 1
    # =========================
    if step_corrente == 1:
        # ✅ prende tipologia da GPT se presente (supporta 2 chiavi), altrimenti fallback
        tipologia_gpt = (data.get("tipologia_stimata") or data.get("tipologia") or "").strip()
        tipologia_finale = tipologia_gpt or tipologia

        db[analisi_col].update_one(
            {"_id": oid},
            {"$set": {
                "marca_stimata": data.get("marca_stimata"),
                "modello_stimato": data.get("modello_stimato"),
                "tipologia": tipologia_finale  # ✅ aggiunto
            }}
        )

        # ✅ importantissimo: aggiorna anche la variabile locale subito
        tipologia = tipologia_finale

        # 🔥 VADEMECUM — SOLO ORA HA SENSO
        vademecum_text = ""
        vmeta = {"found": False, "reason": "not_resolved"}

        brand_raw = data.get("marca_stimata")
        model_raw = data.get("modello_stimato")

        logger.info(f"[VADEMECUM CALL] brand='{brand_raw}' model='{model_raw}'")

        if brand_raw and model_raw:
            vademecum_text, vmeta = load_vademecum_mongo(
                model_raw,
                brand_raw,
                db
            )

    # =========================
    # 7) COSTRUISCI JSON RESPONSE COMPLETO (come versione MySQL)
    # =========================
    json_response_full = {
        **data,
        "id_analisi": str(oid),
        "step": step_corrente,
        "tot_foto": num_foto,

        # ✅ aggiunto: tipologia stabile nel JSON
        "tipologia": tipologia,

        "prompt_info": {
            "prompt_name": meta_prompt.get("prompt_name"),
            "prompt_version": meta_prompt.get("version"),
            "user_id": meta_prompt.get("user_id"),
            "tipologia": tipologia,  # ✅ ora coerente
            "prompt_char_len": len(meta_prompt.get("content", "")),
        },
        "vademecum_info": vmeta,
        "vademecum_text": vademecum_text,
        "timing": timing
    }

    db[foto_col].update_one(
        {"id_analisi": oid, "step": step_corrente},
        {"$set": {"json_response": json_response_full}}
    )

    # ======================================================
    # 🔒 HARD STOP: se siamo all’ultima foto, chiudi comunque
    # ======================================================
    if step_corrente >= MAX_FOTO:
        data["richiedi_altra_foto"] = False
        data["note_backend"] = f"Giudizio finale forzato alla foto {step_corrente}"

    # =========================
    # 8) AGGIORNA STATO ANALISI
    # =========================
    richiedi = data.get("richiedi_altra_foto")

    if richiedi is False:
        # 🔒 STEP FINALE — scriviamo il riepilogo in aut_analisi
        db[analisi_col].update_one(
            {"_id": oid},
            {"$set": {
                "stato": "completata",
                "step_corrente": step_corrente,
                "percentuale_contraffazione": data.get("percentuale"),
                "giudizio_finale": data.get("motivazione"),
                "completed_at": datetime.now(timezone.utc)
            }}
        )
    else:
        # 🟡 STEP INTERMEDIO
        db[analisi_col].update_one(
            {"_id": oid},
            {"$set": {
                "stato": "in_corso",
                "step_corrente": step_corrente
            }}
        )

    return json_safe(json_response_full)


@app.get("/stato-analisi/{id_analisi}")
def stato_analisi(id_analisi: str):
    db = get_db()

    try:
        oid = ObjectId(id_analisi)
    except Exception:
        raise HTTPException(status_code=400, detail="id_analisi non valido")

    analisi = db[analisi_col].find_one({"_id": oid})
    if not analisi:
        raise HTTPException(status_code=404, detail="Analisi non trovata")

    analisi_out = {
        "id_analisi": str(analisi["_id"]),
        "user_id": analisi.get("user_id"),
        "stato": analisi.get("stato"),
        "step_corrente": analisi.get("step_corrente"),
        # ✅ aggiunto
        "tipologia": analisi.get("tipologia"),
        "marca_stimata": analisi.get("marca_stimata"),
        "modello_stimato": analisi.get("modello_stimato"),
        "percentuale_contraffazione": analisi.get("percentuale_contraffazione"),
        "giudizio_finale": analisi.get("giudizio_finale"),
        "created_at": to_dt_utc(analisi.get("created_at")),
    }

    rows = list(
        db[foto_col]
          .find({"id_analisi": oid}, {"_id": 0})
          .sort("step", 1)
    )

    immagini = []
    foto = []
    ultimo_json = None

    for r in rows:
        if r.get("foto_base64"):
            immagini.append(r["foto_base64"])

        jr = r.get("json_response")
        if isinstance(jr, dict):
            ultimo_json = jr

        foto.append({
            "step": r.get("step"),
            "json_response": jr
        })

    if ultimo_json is None:
        ultimo_json = {}

    return {
        "analisi": analisi_out,
        "immagini_base64": immagini,
        "foto": foto,
        "ultimo_json": ultimo_json
    }


@app.get("/admin/analisi")
def admin_list_analisi(
    search: Optional[str] = Query(None),
    stato: Optional[str] = Query(None)
):
    db = get_db()

    match = {}

    if stato:
        match["stato"] = stato

    if search:
        regex = {"$regex": search, "$options": "i"}
        match["$or"] = [
            {"user_id": regex},
            {"marca_stimata": regex},
            {"modello_stimato": regex},
            {"_id": ObjectId(search)} if ObjectId.is_valid(search) else {"_id": None}
        ]

    pipeline = [
        {"$match": match},
        {
            "$lookup": {
                "from": "aut_analisi_foto",
                "localField": "_id",
                "foreignField": "id_analisi",
                "as": "foto"
            }
        },
        {
            "$addFields": {
                "totale_foto": {"$size": "$foto"},
                "last_step": {"$max": "$foto.step"}
            }
        },
        {
            "$project": {
                "foto": 0
            }
        },
        {
            "$sort": {"created_at": -1}
        }
    ]

    rows = list(db.aut_analisi.aggregate(pipeline))

    out = []
    for r in rows:
        out.append({
            "id": str(r["_id"]),
            "user_id": r.get("user_id"),
            "stato": r.get("stato"),
            "step_corrente": r.get("step_corrente"),
            # ✅ aggiunto
            "tipologia": r.get("tipologia"),
            "marca_stimata": r.get("marca_stimata"),
            "modello_stimato": r.get("modello_stimato"),
            "percentuale_contraffazione": r.get("percentuale_contraffazione"),
            "totale_foto": r.get("totale_foto", 0),
            "last_step": r.get("last_step") or 1,
            "created_at": safe_iso_datetime(r.get("created_at")) if r.get("created_at") else None
        })

    return out


@app.get("/admin/analisi/{id}")
def admin_analisi_dettaglio(id: str):
    db = get_db()

    try:
        oid = ObjectId(id)
    except Exception:
        raise HTTPException(status_code=400, detail="ID non valido")

    analisi = db.aut_analisi.find_one({"_id": oid})
    if not analisi:
        raise HTTPException(status_code=404, detail="Analisi non trovata")

    foto = list(
        db.aut_analisi_foto
          .find({"id_analisi": oid})
          .sort("step", 1)
    )

    def safe_dt(x):
        from datetime import datetime, timezone
        if isinstance(x, datetime):
            return x.isoformat()
        if isinstance(x, (int, float)):
            return datetime.fromtimestamp(x, tz=timezone.utc).isoformat()
        return None

    analisi_out = {
        "id": str(analisi["_id"]),
        "user_id": analisi.get("user_id"),
        "stato": analisi.get("stato"),
        "step_corrente": analisi.get("step_corrente"),
        # ✅ aggiunto
        "tipologia": analisi.get("tipologia"),
        "marca_stimata": analisi.get("marca_stimata"),
        "modello_stimato": analisi.get("modello_stimato"),
        "percentuale_contraffazione": analisi.get("percentuale_contraffazione"),
        "giudizio_finale": analisi.get("giudizio_finale"),
        "created_at": safe_dt(analisi.get("created_at")),
        "completed_at": safe_dt(analisi.get("completed_at"))
    }

    foto_out = []
    for f in foto:
        foto_out.append({
            "id": str(f["_id"]),
            "step": f.get("step"),
            "foto_base64": f.get("foto_base64"),
            "json_response": f.get("json_response"),
            "created_at": safe_dt(f.get("created_at"))
        })

    return {
        "analisi": analisi_out,
        "foto": foto_out
    }


# ============================================
# LOGIN
# ============================================

class LoginInput(BaseModel):
    user_id: str
    password: str
    
def is_bcrypt_hash(h: str) -> bool:
    return h.startswith("$2a$") or h.startswith("$2b$") or h.startswith("$2y$")

def verify_legacy_password(plain: str, legacy_hash: str) -> bool:
    """
    Verifica password legacy (MySQL-style).
    Qui assumiamo MD5, che è il caso più comune.
    Se sai che è SHA1 o altro, lo adattiamo.
    """
    md5 = hashlib.md5(plain.encode("utf-8")).hexdigest()
    return md5 == legacy_hash


@app.post("/auth/login")
def auth_login(payload: LoginInput):
    db = get_db()

    user_id = (payload.user_id or "").strip()
    password = payload.password or ""

    if not user_id or not password:
        raise HTTPException(status_code=400, detail="Dati mancanti")

    user = db.aut_users.find_one({
        "user_id": user_id,
        "is_active": True
    })

    if not user:
        raise HTTPException(status_code=401, detail="Credenziali non valide")

    stored_hash = user.get("password_hash")
    if not stored_hash:
        raise HTTPException(status_code=401, detail="Credenziali non valide")

    authenticated = False
    upgraded = False

    # ======================================================
    # 1️⃣ BCRYPT
    # ======================================================
    if is_bcrypt_hash(stored_hash):
        try:
            authenticated = bcrypt.checkpw(
                password.encode("utf-8"),
                stored_hash.encode("utf-8")
            )
        except Exception:
            authenticated = False

    # ======================================================
    # 2️⃣ LEGACY (MD5)
    # ======================================================
    else:
        authenticated = verify_legacy_password(password, stored_hash)

        # upgrade automatico
        if authenticated:
            new_hash = bcrypt.hashpw(
                password.encode("utf-8"),
                bcrypt.gensalt()
            ).decode("utf-8")

            db.aut_users.update_one(
                {"_id": user["_id"]},
                {
                    "$set": {
                        "password_hash": new_hash,
                        "must_reset_password": False,
                        "password_upgraded_at": datetime.now(timezone.utc)
                    }
                }
            )
            upgraded = True

    if not authenticated:
        raise HTTPException(status_code=401, detail="Credenziali non valide")

    # ======================================================
    # 3️⃣ AUDIT
    # ======================================================
    try:
        db.aut_login_log.insert_one({
            "user_id": user_id,
            "ts": datetime.now(timezone.utc),
            "upgraded": upgraded
        })
    except Exception:
        pass

    # ======================================================
    # 4️⃣ RESPONSE
    # ======================================================
    return {
        "success": True,
        "user_id": user_id,
        "role": user.get("role", "user"),
        "password_upgraded": upgraded,
        "must_reset_password": user.get("must_reset_password", False)
    }

# ============================================
# HEALTHCHECK
# ============================================
@app.get("/")
def root():
    uri = os.getenv("MONGO_URI", "mongodb+srv://appl_cora_llm:appl_cora_llm@svi01-mngdb-svil.sogei.it/appl_cora_llm?tls=false")
    if not uri:
        return {"status": "error", "error": "MONGO_URI missing"}
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000, retryWrites=False)
        client.admin.command("ping")
        return {"status": "ok", "host": client.address[0], "port": client.address[1], "db": MONGO_DB_NAME}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ==========================
# VADEMECUM CRUD - MODELS
# ==========================
class VademecumCreate(BaseModel):
    type: str = "model"     # "model" | "brand_generic" | "general"
    brand: str
    model: str = ""         # per "general" può restare ""
    raw_text: str

class VademecumUpdate(BaseModel):
    type: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    raw_text: Optional[str] = None

def _norm(s: str) -> str:
    return normalize(s or "")

def _vademecum_out(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(doc["_id"]),
        "type": doc.get("type"),
        "brand": doc.get("brand"),
        "brand_norm": doc.get("brand_norm"),
        "model": doc.get("model"),
        "model_norm": doc.get("model_norm"),
        "raw_text": doc.get("raw_text", ""),
        "created_at": safe_iso_datetime(doc.get("created_at")),
        "updated_at": safe_iso_datetime(doc.get("updated_at")),
    }

def _duplicate_exists(col, *, type_: str, brand_norm: str, model_norm: str, exclude_id: Optional[ObjectId] = None) -> bool:
    q = {"type": type_, "brand_norm": brand_norm, "model_norm": model_norm}
    if exclude_id is not None:
        q["_id"] = {"$ne": exclude_id}
    return col.count_documents(q, limit=1) > 0


# ==========================
# VADEMECUM CRUD - ENDPOINTS
# ==========================

@app.get("/admin/vademecum")
def admin_vademecum_list(
    q: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    skip: int = Query(0, ge=0),
):
    db = get_db()
    col = db[vademecum_col]

    match: Dict[str, Any] = {}

    if type:
        match["type"] = type.strip()

    if brand:
        match["brand_norm"] = _norm(brand)

    if model is not None and model != "":
        match["model_norm"] = _norm(model)

    if q:
        rx = {"$regex": q, "$options": "i"}
        match["$or"] = [{"brand": rx}, {"model": rx}, {"raw_text": rx}, {"type": rx}]

    rows = list(
        col.find(match)
           .sort([("brand_norm", 1), ("model_norm", 1)])
           .skip(skip)
           .limit(limit)
    )

    return {"count": col.count_documents(match), "items": [_vademecum_out(r) for r in rows]}


@app.get("/admin/vademecum/{id}")
def admin_vademecum_get(id: str):
    db = get_db()
    col = db[vademecum_col]

    if not ObjectId.is_valid(id):
        raise HTTPException(status_code=400, detail="ID non valido")

    doc = col.find_one({"_id": ObjectId(id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Vademecum non trovato")

    return _vademecum_out(doc)


@app.post("/admin/vademecum")
def admin_vademecum_create(payload: VademecumCreate):
    db = get_db()
    col = db[vademecum_col]

    type_ = (payload.type or "model").strip()
    brand = (payload.brand or "").strip()
    model = (payload.model or "").strip()
    raw_text = (payload.raw_text or "").strip()

    if not brand:
        raise HTTPException(status_code=400, detail="brand obbligatorio")
    if not raw_text:
        raise HTTPException(status_code=400, detail="raw_text obbligatorio")

    brand_norm = _norm(brand)
    model_norm = _norm(model)

    if _duplicate_exists(col, type_=type_, brand_norm=brand_norm, model_norm=model_norm):
        raise HTTPException(status_code=409, detail="Duplicato: stesso type/brand/model")

    now = datetime.now(timezone.utc)
    doc = {
        "type": type_,
        "brand": brand,
        "brand_norm": brand_norm,
        "model": model,
        "model_norm": model_norm,
        "raw_text": raw_text,
        "created_at": now,
        "updated_at": now,
    }

    res = col.insert_one(doc)
    doc["_id"] = res.inserted_id
    return _vademecum_out(doc)


@app.put("/admin/vademecum/{id}")
def admin_vademecum_update(id: str, payload: VademecumUpdate):
    db = get_db()
    col = db[vademecum_col]

    if not ObjectId.is_valid(id):
        raise HTTPException(status_code=400, detail="ID non valido")
    oid = ObjectId(id)

    cur = col.find_one({"_id": oid})
    if not cur:
        raise HTTPException(status_code=404, detail="Vademecum non trovato")

    type_ = (payload.type.strip() if isinstance(payload.type, str) and payload.type.strip() else cur.get("type", "model"))
    brand = (payload.brand.strip() if isinstance(payload.brand, str) and payload.brand.strip() else cur.get("brand", ""))
    model = (payload.model.strip() if isinstance(payload.model, str) else cur.get("model", ""))
    raw_text = (payload.raw_text.strip() if isinstance(payload.raw_text, str) else cur.get("raw_text", ""))

    if not brand:
        raise HTTPException(status_code=400, detail="brand obbligatorio")
    if not raw_text:
        raise HTTPException(status_code=400, detail="raw_text obbligatorio")

    brand_norm = _norm(brand)
    model_norm = _norm(model)

    if _duplicate_exists(col, type_=type_, brand_norm=brand_norm, model_norm=model_norm, exclude_id=oid):
        raise HTTPException(status_code=409, detail="Duplicato: stesso type/brand/model")

    upd = {
        "type": type_,
        "brand": brand,
        "brand_norm": brand_norm,
        "model": model,
        "model_norm": model_norm,
        "raw_text": raw_text,
        "updated_at": datetime.now(timezone.utc),
    }

    col.update_one({"_id": oid}, {"$set": upd})
    doc = col.find_one({"_id": oid})
    return _vademecum_out(doc)


@app.delete("/admin/vademecum/{id}")
def admin_vademecum_delete(id: str):
    db = get_db()
    col = db[vademecum_col]

    if not ObjectId.is_valid(id):
        raise HTTPException(status_code=400, detail="ID non valido")

    oid = ObjectId(id)
    doc = col.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Vademecum non trovato")

    col.delete_one({"_id": oid})
    return {"status": "ok", "deleted_id": id}

# In[ ]:


# # ======================================================
# # MAIN SERVER
# # ======================================================
# if __name__ == "__main__":
#     config = uvicorn.Config(app, host="127.0.0.1",port=8077)
#     server = uvicorn.Server(config)
#     await server.serve()


# In[ ]:


# import os
# import mysql.connector
# from pymongo import MongoClient
# from datetime import datetime
# import json

# # ======================================================
# # MYSQL (come da tuo backend)
# # ======================================================
# def get_mysql_connection():
#     return mysql.connector.connect(
#         host="127.0.0.1",
#         user="root",
#         password="",
#         database="autentica",
#         use_pure=True
#     )

# # ======================================================
# # MONGO
# # ======================================================
# MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://appl_tssanita:appl_tssanita@svi02-mngdb-svil.sogei.it/appl_tssanita?tls=false")
# MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "tssanita")


# mongo = MongoClient(MONGO_URI)
# db = mongo[MONGO_DB_NAME]

# # ======================================================
# # MYSQL CONNECT
# # ======================================================
# mysql = get_mysql_connection()
# cur = mysql.cursor(dictionary=True)

# # ======================================================
# # 1️⃣ MIGRAZIONE ANALISI
# # ======================================================
# print("▶ Migrating analisi...")
# analisi_map = {}  # legacy_id (MySQL) -> ObjectId (Mongo)

# cur.execute("SELECT * FROM analisi")
# for row in cur.fetchall():
#     doc = {
#         "legacy_id": row["id"],
#         "user_id": row["user_id"],
#         "stato": row["stato"],
#         "step_corrente": row["step_corrente"],
#         "marca_stimata": row["marca_stimata"],
#         "modello_stimato": row["modello_stimato"],
#         "percentuale_contraffazione": row["percentuale_contraffazione"],
#         "giudizio_finale": row["giudizio_finale"],
#         "created_at": row.get("created_at", datetime.utcnow())
#     }
#     res = db.aut_analisi.insert_one(doc)
#     analisi_map[row["id"]] = res.inserted_id

# print("✔ analisi migrate")

# # ======================================================
# # 2️⃣ MIGRAZIONE ANALISI_FOTO
# # ======================================================
# print("▶ Migrating analisi_foto...")

# cur.execute("SELECT * FROM analisi_foto ORDER BY id_analisi, step")
# for row in cur.fetchall():
#     doc = {
#         "analisi_id": analisi_map[row["id_analisi"]],
#         "legacy_id_analisi": row["id_analisi"],
#         "step": row["step"],
#         "foto_base64": row["foto_base64"],
#         "json_response": json.loads(row["json_response"]) if row["json_response"] else None,
#         "created_at": row.get("created_at", datetime.utcnow())
#     }
#     db.aut_analisi_foto.insert_one(doc)

# print("✔ analisi_foto migrate")

# # ======================================================
# # 3️⃣ MIGRAZIONE PROMPTS
# # ======================================================
# print("▶ Migrating prompts...")

# cur.execute("SELECT * FROM prompts")
# for row in cur.fetchall():
#     db.aut_prompts.insert_one({
#         "legacy_id": row["id"],
#         "name": row["name"],
#         "created_at": row["created_at"]
#     })

# print("✔ prompts migrate")

# # ======================================================
# # 4️⃣ MIGRAZIONE PROMPT_VERSIONS
# # ======================================================
# print("▶ Migrating prompt_versions...")

# cur.execute("SELECT * FROM prompt_versions")
# for row in cur.fetchall():
#     db.aut_prompt_versions.insert_one({
#         "prompt_name": row["prompt_name"],
#         "user_id": row["user_id"],
#         "version": row["version"],
#         "content": row["content"],
#         "is_active": bool(row["is_active"]),
#         "created_at": row["created_at"]
#     })

# print("✔ prompt_versions migrate")

# # ======================================================
# # CLEANUP
# # ======================================================
# cur.close()
# mysql.close()
# mongo.close()

# print("\n✅ MIGRAZIONE COMPLETATA CON SUCCESSO")


# In[ ]:


# import mysql.connector
# from pymongo import MongoClient
# from datetime import datetime, timezone
# import os
# # ===============================
# # MYSQL CONFIG
# # ===============================
# # MYSQL (come da tuo backend)
# # ======================================================
# def get_mysql_connection():
#     return mysql.connector.connect(
#         host="127.0.0.1",
#         user="root",
#         password="",
#         database="aigov",
#         use_pure=True
#     )

# MYSQL_TABLE = "anagrafica_personale"

# # ===============================
# # MONGO CONFIG
# # ===============================
# MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://appl_tssanita:appl_tssanita@svi02-mngdb-svil.sogei.it/appl_tssanita?tls=false")
# MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "tssanita")
# MONGO_COLLECTION = "aut_users"


# def resolve_role(row: dict) -> str:
#     """
#     Determina il ruolo in modo coerente
#     """
#     if row.get("qualifica"):
#         return row["qualifica"]

#     if row.get("admin") == 1:
#         return "admin"
#     if row.get("viewer") == 1:
#         return "viewer"
#     if row.get("member") == 1:
#         return "member"

#     return "external"


# def main():
#     print("🔄 Migrazione utenti da MySQL → MongoDB")

#     # MYSQL
#     mysql = get_mysql_connection()
#     cur = mysql.cursor(dictionary=True)

#     cur.execute(f"SELECT * FROM {MYSQL_TABLE}")
#     rows = cur.fetchall()

#     if not rows:
#         print("⚠️ Nessun utente trovato")
#         return

#     # MONGO
#     mongo = MongoClient(MONGO_URI)
#     db = mongo[MONGO_DB_NAME]
#     col = db[MONGO_COLLECTION]

#     migrated = 0
#     skipped = 0

#     for r in rows:
#         user_id = r.get("userid")

#         if not user_id:
#             skipped += 1
#             continue

#         # evita duplicati
#         if col.find_one({"user_id": user_id}):
#             print(f"⚠️ già presente: {user_id}")
#             skipped += 1
#             continue

#         password_hash = r.get("password")
#         if not password_hash:
#             print(f"⛔ password mancante per {user_id}")
#             skipped += 1
#             continue

#         doc = {
#             "user_id": user_id,
#             "password_hash": password_hash,  # già hashata
#             "is_active": bool(r.get("fl_attivo", 1)),
#             "role": resolve_role(r),

#             "profile": {
#                 "nome": r.get("nome"),
#                 "cognome": r.get("cognome")
#             },

#             "email": r.get("email"),
#             "phone": r.get("phone"),
#             "home": r.get("home"),

#             "must_reset_password": bool(r.get("reset_password", 1)),

#             "created_at": datetime.now(timezone.utc),
#             "source": "mysql_anagrafica_personale"
#         }

#         col.insert_one(doc)
#         migrated += 1
#         print(f"✅ migrato: {user_id}")

#     print("\n===============================")
#     print("✔️ MIGRAZIONE COMPLETATA")
#     print(f"   Migrati : {migrated}")
#     print(f"   Saltati : {skipped}")
#     print("===============================")

#     cur.close()
#     mysql.close()
#     mongo.close()


# if __name__ == "__main__":
#     main()


# In[ ]:











