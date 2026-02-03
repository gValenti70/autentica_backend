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
from math import ceil


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
            socketTimeoutMS=120000,
            connectTimeoutMS=20000,
            serverSelectionTimeoutMS=20000,
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



def invalido(x: str) -> bool:
    return not x or x.lower() in {
        "non identificabile",
        "non determinabile",
        "sconosciuto",
        "unknown",
        "nd",
        "n.d."
    }

def sostanzialmente_diverso(a: str, b: str) -> bool:
    if not a or not b:
        return True
    return similarity(a.lower(), b.lower()) < 0.60


# # ======================================================
# # VADEMECUM
# # ======================================================


vlog = logging.getLogger("VADEMECUM")
vlog.setLevel(logging.INFO)

BRAND_FUZZY_MIN = 0.85
MODEL_FUZZY_MIN = 0.60  # puoi alzarlo (es 0.75) se vuoi meno false positive
GENERIC_MODEL_NORM = "generico"  # deve combaciare con come lo salvi in Cosmos

def _tokens_norm(s: str):
    tokens = re.sub(r"[^a-zA-Z0-9\s]", " ", (s or "").lower()).split()
    return [normalize(t) for t in tokens if len(t) >= 2], tokens

def _best_fuzzy_match(probes, candidates, field_norm: str):
    best = None
    best_score = 0.0
    best_probe = None

    for c in candidates:
        c_norm = (c.get(field_norm, "") or "")
        for p in probes:
            if not p:
                continue
            score = similarity(p, c_norm)
            if score > best_score:
                best = c
                best_score = score
                best_probe = p

    return best, best_score, best_probe

def load_vademecum_mongo(model: str, brand: str, db):
    meta = {
        "brand_requested": brand or "",
        "model_requested": model or "",
        "brand_norm_in": normalize(brand or ""),
        "model_norm_in": normalize(model or ""),
        "resolved_brand_norm": None,
        "resolved_brand": None,
        "resolved_model_norm": None,
        "resolved_model": None,
        "source": None,
        "match_type": None,
        "vademecum_id": None,
        "fuzzy_score_brand": None,
        "fuzzy_score_model": None,
        "length_chars": None,
        "debug": {}
    }

    col = db["aut_vademecum"]

    brand_raw = brand or ""
    model_raw = model or ""
    brand_norm = meta["brand_norm_in"]
    model_norm = meta["model_norm_in"]

    model_tokens_norm, model_tokens_raw = _tokens_norm(model_raw)

    vlog.info("========== VADEMECUM LOOKUP ==========")
    vlog.info(f"[IN ] brand_raw='{brand_raw}' model_raw='{model_raw}'")
    vlog.info(f"[NORM] brand_norm='{brand_norm}' model_norm='{model_norm}'")
    vlog.info(f"[TOK] tokens={model_tokens_raw}")
    vlog.info(f"[TOK] tokens_norm={model_tokens_norm}")

    # ======================================================
    # A) RISOLVI BRAND (exact -> fuzzy >= 0.85)
    # ======================================================
    resolved_brand_norm = None
    resolved_brand_label = None

    if brand_norm:
        # A1) exact brand_norm su documenti type=model (basta che esista almeno un record per quel brand)
        try:
            exists = col.find_one(
                {"brand_norm": brand_norm, "type": "model"},
                {"brand_norm": 1, "brand": 1}
            )
        except Exception as e:
            vlog.error(f"[BRAND exact] find_one error: {e}")
            exists = None
            meta["debug"]["brand_exact_error"] = str(e)

        if exists:
            resolved_brand_norm = brand_norm
            resolved_brand_label = exists.get("brand") or brand_raw
            meta["match_type"] = "brand_exact"
            vlog.info(f"[BRAND exact] HIT brand_norm='{resolved_brand_norm}'")
        else:
            vlog.info("[BRAND exact] MISS -> fuzzy search")

            # A2) fuzzy: prendi tutti i brand disponibili (type=model) e fai similarity
            try:
                # prendo solo campi utili e DISTINCT-like lato app (Cosmos/Mongo non sempre perfetto con distinct)
                brand_candidates = list(col.find({"type": "model"}, {"brand_norm": 1, "brand": 1}))
            except Exception as e:
                vlog.error(f"[BRAND fuzzy] find error: {e}")
                brand_candidates = []
                meta["debug"]["brand_fuzzy_candidates_error"] = str(e)

            # dedup per brand_norm
            seen = set()
            uniq = []
            for c in brand_candidates:
                bn = c.get("brand_norm")
                if not bn or bn in seen:
                    continue
                seen.add(bn)
                uniq.append(c)

            best, best_score, _ = _best_fuzzy_match([brand_norm], uniq, "brand_norm")

            vlog.info(f"[BRAND fuzzy] best_score={best_score:.3f} best='{(best or {}).get('brand_norm')}'")

            if best and best_score >= BRAND_FUZZY_MIN:
                resolved_brand_norm = best.get("brand_norm")
                resolved_brand_label = best.get("brand") or brand_raw
                meta["match_type"] = "brand_fuzzy"
                meta["fuzzy_score_brand"] = round(best_score, 3)
                vlog.info(f"[BRAND fuzzy] ACCEPT brand_norm='{resolved_brand_norm}'")
            else:
                vlog.info("[BRAND] NOT RESOLVED (no exact, fuzzy < threshold) -> return empty")
                meta["source"] = "no_brand"
                return "", meta

    else:
        meta["source"] = "no_brand"
        return "", meta

    meta["resolved_brand_norm"] = resolved_brand_norm
    meta["resolved_brand"] = resolved_brand_label

    # ======================================================
    # B) RISOLVI MODELLO (exact -> fuzzy; se no -> GENERICO)
    # ======================================================
    def _fetch_model_exact(m_norm: str):
        if not m_norm:
            return None
        try:
            return col.find_one(
                {"brand_norm": resolved_brand_norm, "model_norm": m_norm, "type": "model"},
                {"raw_text": 1, "model_norm": 1, "model": 1}
            )
        except Exception as e:
            vlog.error(f"[MODEL exact] find_one error: {e}")
            meta["debug"]["model_exact_error"] = str(e)
            return None

    # B1) exact full model_norm
    doc = _fetch_model_exact(model_norm)

    # B1b) exact token (prime 10)
    if not doc and model_tokens_norm:
        for t in model_tokens_norm[:10]:
            doc = _fetch_model_exact(t)
            if doc:
                meta["debug"]["model_exact_token_hit"] = t
                break

    if doc and doc.get("raw_text"):
        text = doc["raw_text"]
        meta.update({
            "source": "model_exact",
            "vademecum_id": str(doc.get("_id")),
            "resolved_model_norm": doc.get("model_norm"),
            "resolved_model": doc.get("model"),
            "length_chars": len(text)
        })
        return text, meta

    # B2) fuzzy model dentro brand
    probes = ([model_norm] if model_norm else []) + model_tokens_norm[:10]
    if probes:
        try:
            candidates = list(col.find(
                {"brand_norm": resolved_brand_norm, "type": "model"},
                {"model_norm": 1, "model": 1, "raw_text": 1}
            ))
        except Exception as e:
            vlog.error(f"[MODEL fuzzy] find error: {e}")
            candidates = []
            meta["debug"]["model_fuzzy_candidates_error"] = str(e)

        best, best_score, best_probe = _best_fuzzy_match(probes, candidates, "model_norm")
        vlog.info(f"[MODEL fuzzy] best_score={best_score:.3f} best_probe='{best_probe}' best='{(best or {}).get('model_norm')}'")

        if best and best_score >= MODEL_FUZZY_MIN and best.get("raw_text"):
            text = best["raw_text"]
            meta.update({
                "source": "model_fuzzy",
                "vademecum_id": str(best.get("_id")),
                "resolved_model_norm": best.get("model_norm"),
                "resolved_model": best.get("model"),
                "fuzzy_score_model": round(best_score, 3),
                "length_chars": len(text)
            })
            return text, meta

    # B3) modello GENERICO del brand
    doc = _fetch_model_exact(GENERIC_MODEL_NORM)
    if doc and doc.get("raw_text"):
        text = doc["raw_text"]
        meta.update({
            "source": "model_generic",
            "vademecum_id": str(doc.get("_id")),
            "resolved_model_norm": doc.get("model_norm"),
            "resolved_model": doc.get("model") or "Generico",
            "length_chars": len(text)
        })
        return text, meta

    # ======================================================
    # C) NIENTE GENERICO -> lascia decidere OpenAI
    # ======================================================
    meta["source"] = "openai"
    meta["match_type"] = meta.get("match_type") or "brand_resolved_no_model"
    return "", meta




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
    # 6) Gestione marca/modello (step 1 + possibile revisione step 2)
    # =========================
    
    marca_db = (analisi or {}).get("marca_stimata")
    modello_db = (analisi or {}).get("modello_stimato")
    
    marca_new = (data.get("marca_stimata") or "").strip()
    modello_new = (data.get("modello_stimato") or "").strip()
        
    update = {}
    rifai_vademecum = False
    
        
    # -------------------------
    # STEP 1 → salva sempre
    # -------------------------
    if step_corrente == 1:
        if marca_new:
            update["marca_stimata"] = marca_new
        if modello_new:
            update["modello_stimato"] = modello_new
    
    # -------------------------
    # STEP 2 → OVERRIDE FORTE
    # -------------------------
    elif step_corrente == 2:
        cambio_marca = (
            not invalido(marca_new)
            and sostanzialmente_diverso(marca_db or "", marca_new)
        )
    
        cambio_modello = (
            not invalido(modello_new)
            and sostanzialmente_diverso(modello_db or "", modello_new)
        )
    
        if cambio_marca or cambio_modello:
            update["marca_stimata"] = marca_new
            update["modello_stimato"] = modello_new
            rifai_vademecum = True
    
    # -------------------------
    # Tipologia → SOLO step 1
    # -------------------------
    if step_corrente == 1:
        tipologia_gpt = (data.get("tipologia_stimata") or data.get("tipologia") or "").strip()
        if tipologia_gpt:
            update["tipologia"] = tipologia_gpt
    
    # -------------------------
    # Scrittura DB
    # -------------------------
    if update:
        db[analisi_col].update_one(
            {"_id": oid},
            {"$set": update}
        )
    
    # -------------------------
    # Rifai vademecum se override
    # -------------------------
    if rifai_vademecum:
        logger.info(
            f"[VADEMECUM OVERRIDE] nuova marca='{marca_new}' nuovo modello='{modello_new}'"
        )
        vademecum_text, vmeta = load_vademecum_mongo(
            modello_new,
            marca_new,
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
    stato: Optional[str] = Query(None),
    marca: Optional[str] = Query(None),
    modello: Optional[str] = Query(None),
    min_percentuale: Optional[int] = Query(None, ge=0, le=100),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=200),
):
    db = get_db()

    match = {}

    # =========================
    # FILTRI ESATTI (dashboard)
    # =========================
    if stato:
        match["stato"] = stato.lower()

    if marca:
        match["marca_stimata"] = marca

    if modello:
        match["modello_stimato"] = modello

    # =========================
    # FILTRO PERCENTUALE (string → int)
    # =========================
    if min_percentuale is not None:
        match["$expr"] = {
            "$gte": [
                { "$toInt": "$percentuale_contraffazione" },
                min_percentuale
            ]
        }

    # =========================
    # SEARCH LIBERA (admin)
    # =========================
    if search:
        regex = {"$regex": search, "$options": "i"}
        or_list = [
            {"user_id": regex},
            {"marca_stimata": regex},
            {"modello_stimato": regex},
        ]
        if ObjectId.is_valid(search):
            or_list.append({"_id": ObjectId(search)})
        match["$or"] = or_list

    # =========================
    # PAGINAZIONE
    # =========================
    total = db.aut_analisi.count_documents(match)
    total_pages = max(1, ceil(total / page_size))

    if page > total_pages:
        page = total_pages

    skip = (page - 1) * page_size

    # =========================
    # PIPELINE
    # =========================
    pipeline = [
        {"$match": match},
        {"$sort": {"created_at": -1}},
        {"$skip": skip},
        {"$limit": page_size},
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
                "last_step": {
                    "$ifNull": [
                        {"$max": "$foto.step"},
                        1
                    ]
                }
            }
        },
        {"$project": {"foto": 0}}
    ]

    rows = list(db.aut_analisi.aggregate(pipeline))

    # =========================
    # OUTPUT
    # =========================
    items = []
    for r in rows:
        items.append({
            "id": str(r["_id"]),
            "user_id": r.get("user_id"),
            "stato": r.get("stato"),
            "step_corrente": r.get("step_corrente"),
            "tipologia": r.get("tipologia"),
            "marca_stimata": r.get("marca_stimata"),
            "modello_stimato": r.get("modello_stimato"),
            "percentuale_contraffazione": r.get("percentuale_contraffazione"),
            "totale_foto": r.get("totale_foto", 0),
            "last_step": r.get("last_step") or 1,
            "created_at": safe_iso_datetime(r.get("created_at"))
        })

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages
    }




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

from typing import Optional
from fastapi import Query
from math import ceil


@app.get("/admin/dashboard/top-contrafatti")
def admin_dashboard_top_contraffatti(
    soglia: int = Query(50, ge=0, le=99),
    limit: int = Query(5, ge=1, le=50),
):
    db = get_db()

    INVALID_VALUES = [
        None,
        "",
        "nd",
        "n.d.",
        "non determinabile",
        "non identificabile",
        "incerto",
    ]

    pipeline = [
        # 1️⃣ solo analisi finali
        {
            "$match": {
                "stato": "completata",
                "percentuale_contraffazione": {"$ne": None},
                "marca_stimata": {"$nin": INVALID_VALUES},
                "modello_stimato": {"$nin": INVALID_VALUES}
            }
        },

        # 2️⃣ percentuale → int
        {
            "$addFields": {
                "percentuale_num": {
                    "$toInt": "$percentuale_contraffazione"
                }
            }
        },

        # 3️⃣ filtri SEMANTICI
        {
            "$match": {
                "percentuale_num": {
                    "$gte": soglia,
                    "$lt": 100          # ❌ esclude i 100
                }
            }
        },

        # 4️⃣ aggregazione TOP FAKE
        {
            "$group": {
                "_id": {
                    "marca": "$marca_stimata",
                    "modello": "$modello_stimato"
                },
                "tot": { "$sum": 1 },
                "percentuale_media": { "$avg": "$percentuale_num" },
                "percentuale_max": { "$max": "$percentuale_num" },
                "sample_analisi_id": { "$first": "$_id" }
            }
        },

        { "$sort": { "tot": -1, "percentuale_media": -1 } },
        { "$limit": limit },

        # 5️⃣ foto rappresentativa (SOLO step 1)
        {
            "$lookup": {
                "from": "aut_analisi_foto",
                "let": { "aid": "$sample_analisi_id" },
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    { "$eq": ["$id_analisi", "$$aid"] },
                                    { "$eq": ["$step", 1] }
                                ]
                            }
                        }
                    },
                    { "$project": { "_id": 0, "foto_base64": 1 } },
                    { "$limit": 1 }
                ],
                "as": "foto"
            }
        },

        
        # # 5️⃣ foto rappresentativa (step 2 > step 1)
        # {
        #     "$lookup": {
        #         "from": "aut_analisi_foto",
        #         "let": { "aid": "$sample_analisi_id" },
        #         "pipeline": [
        #             {
        #                 "$match": {
        #                     "$expr": {
        #                         "$and": [
        #                             { "$eq": ["$id_analisi", "$$aid"] },
        #                             { "$in": ["$step", [1, 2]] }
        #                         ]
        #                     }
        #                 }
        #             },
        #             { "$sort": { "step": -1 } },
        #             { "$limit": 1 },
        #             { "$project": { "_id": 0, "foto_base64": 1 } }
        #         ],
        #         "as": "foto"
        #     }
        # },

        # 6️⃣ flatten
        {
            "$addFields": {
                "foto": { "$first": "$foto.foto_base64" }
            }
        }
    ]

    rows = list(db.aut_analisi.aggregate(pipeline))

    return [
        {
            "marca": r["_id"]["marca"],
            "modello": r["_id"]["modello"],
            "tot": r["tot"],
            "percentuale_media": round(r["percentuale_media"], 1),
            "percentuale_max": r["percentuale_max"],
            "foto": r.get("foto")
        }
        for r in rows
    ]


@app.get("/admin/dashboard/fake-ratio")
def admin_dashboard_fake_ratio(soglia: int = 50):
    db = get_db()

    total = db.aut_analisi.count_documents({"step_corrente": 1})

    fake = db.aut_analisi.count_documents({
        "step_corrente": 1,
        "$expr": {
            "$lte": [
                { "$toInt": "$percentuale_contraffazione" },
                soglia
            ]
        }
    })

    return {
        "fake": fake,
        "genuine": max(0, total - fake)
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



# # ======================================================
# # MAIN SERVER
# # ======================================================
# if __name__ == "__main__":
#     config = uvicorn.Config(app, host="127.0.0.1",port=8077)
#     server = uvicorn.Server(config)
#     await server.serve()



























