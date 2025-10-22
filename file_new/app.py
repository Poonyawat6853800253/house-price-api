# app.py - FastAPI simple + robust pickle shim + 3 cards with price inverse
import os
import sys
import types
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =========================
# Config
# =========================
MODEL_FILE = os.getenv("MODEL_FILE", "house_price_pipeline_Train_10.joblib")
INT_LIKE = ["OverallQual", "GarageCars", "Fireplaces", "BedroomAbvGr", "FullBath"]

# =========================
# SHIM: ให้ unpickler หา HousePriceModel เจอ (กรณี joblib เซฟด้วยคลาส custom)
# =========================
class HousePriceModel:
    """Minimal shim so that joblib pickle can resolve the symbol."""
    def __init__(self): ...
    def __setstate__(self, state): self.__dict__.update(state)
    def _inner(self):
        for name in ("pipeline", "pipe", "model", "estimator", "clf"):
            if hasattr(self, name):
                return getattr(self, name)
        return None
    def predict(self, X):
        inner = self._inner()
        if inner is None:
            raise RuntimeError("HousePriceModel shim: no inner estimator found (expected: pipeline/pipe/model/estimator/clf)")
        return inner.predict(X)

# ผูกคลาสนี้เข้าไปในโมดูลที่ pickle จะหา
for modname in ("__main__", "__mp_main__"):
    if modname in sys.modules:
        setattr(sys.modules[modname], "HousePriceModel", HousePriceModel)
    else:
        m = types.ModuleType(modname)
        setattr(m, "HousePriceModel", HousePriceModel)
        sys.modules[modname] = m

# =========================
# Utilities
# =========================
def _unwrap_estimator(obj: Any) -> Any:
    """คืน estimator/pipeline ภายใน ถ้ามี; ถ้าไม่มี แปลว่า obj คือ estimator/pipeline อยู่แล้ว"""
    for name in ("pipeline", "pipe", "model", "estimator", "clf"):
        if hasattr(obj, name):
            return getattr(obj, name)
    return obj

def _find_column_transformer(pipeline_like: Any):
    from sklearn.compose import ColumnTransformer
    if isinstance(pipeline_like, ColumnTransformer):
        return pipeline_like
    if hasattr(pipeline_like, "named_steps"):
        for _, step in pipeline_like.named_steps.items():
            if isinstance(step, ColumnTransformer):
                return step
    return None

def extract_defaults_from_pipeline(pipeline_like: Any) -> Dict[str, Any]:
    """ดึง default จาก SimpleImputer (median/mode) ถ้ามี ColumnTransformer"""
    defaults: Dict[str, Any] = {}
    ct = _find_column_transformer(pipeline_like)
    if ct is None:
        return defaults

    from sklearn.impute import SimpleImputer
    for _, trans, cols in ct.transformers_:
        if trans == "drop":
            continue
        imputer: Optional[SimpleImputer] = None
        if hasattr(trans, "steps"):
            for _, est in trans.steps:
                if isinstance(est, SimpleImputer):
                    imputer = est
                    break
        elif isinstance(trans, SimpleImputer):
            imputer = trans

        if imputer is None:
            continue
        stats = getattr(imputer, "statistics_", None)
        if stats is not None and len(stats) == len(cols):
            for c, val in zip(cols, stats):
                defaults[c] = val
    return defaults

def cast_int_like(feat: Dict[str, Any]) -> None:
    """ปัดคอลัมน์ที่ควรเป็นจำนวนเต็มให้เป็น int"""
    for ik in INT_LIKE:
        if ik in feat and isinstance(feat[ik], (int, float, np.integer, np.floating)):
            feat[ik] = int(round(float(feat[ik])))

# =========================
# Load model once
# =========================
if not os.path.exists(MODEL_FILE):
    raise RuntimeError(f"Model file not found: {MODEL_FILE}")

try:
    _loaded = joblib.load(MODEL_FILE)           # ← ตอนนี้ unpickler หา HousePriceModel เจอแล้ว
    _inner = _unwrap_estimator(_loaded)
    PREDICTOR = _loaded if hasattr(_loaded, "predict") else _inner
    PIPE_FOR_DEFAULTS = _inner
    print(f"✅ Loaded model: {MODEL_FILE} | predictor={type(PREDICTOR).__name__} | defaults_from={type(PIPE_FOR_DEFAULTS).__name__}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# =========================
# Heuristic: ตรวจว่าทำนายเป็น log หรือไม่ + inverse
# =========================
def _probe_defaults_for_price_like():
    try:
        d = extract_defaults_from_pipeline(PIPE_FOR_DEFAULTS)
        if not d:
            return None
        X0 = pd.DataFrame([d])
        y0 = float(PREDICTOR.predict(X0)[0])
        return y0
    except Exception:
        return None

def _looks_like_log(v: float) -> bool:
    # ค่าราคาแบบ ln/log1p ทั่วไปอยู่ช่วง ~10–14 (เช่น 12.x)
    return 5.0 < v < 20.0

PROBED_Y0 = _probe_defaults_for_price_like()
PREDICTION_LOOKS_LOG = (_looks_like_log(PROBED_Y0) if PROBED_Y0 is not None else False)

def invert_target(y: float) -> float:
    """
    ถ้าดูเหมือนเป็น log-target → แปลงกลับเป็นราคาจริง
    ปรับให้ตรงกับตอนเทรน:
    - ถ้าเทรนด้วย np.log1p(SalePrice) ⇒ ใช้ np.expm1
    - ถ้าเทรนด้วย np.log(SalePrice)  ⇒ เปลี่ยนเป็น: return float(np.exp(y))
    """
    if _looks_like_log(y) or PREDICTION_LOOKS_LOG:
        return float(np.expm1(y))  # ถ้าใช้ ln แท้ ให้เปลี่ยนเป็น: return float(np.exp(y))
    return float(y)

# =========================
# Core logic: สร้าง 3 cards
# =========================
def make_three_cards(fixed: Dict[str, Any], pct_window: float = 0.15, usd_to_thb: float = 36.0) -> List[Dict[str, Any]]:
    if PREDICTOR is None or PIPE_FOR_DEFAULTS is None:
        raise RuntimeError("Model not loaded")

    defaults = extract_defaults_from_pipeline(PIPE_FOR_DEFAULTS)
    if not defaults:
        # Fallback เมื่อไม่มี imputer/statistics ให้ดึงจริง
        defaults = {
            "OverallQual": 6,
            "TotalBsmtSF": 800,
            "LotArea": 9000,
            "GarageCars": 2,
            "Fireplaces": 1,
            "BedroomAbvGr": 3,
            "GrLivArea": 1500,
            "FullBath": 2,
            "Neighborhood": "NAmes",
        }

    # ฐานจาก defaults แล้วทับด้วยค่าที่ผู้ใช้ล็อค
    base = defaults.copy()
    for k, v in (fixed or {}).items():
        base[k] = v

    # ปรับเฉพาะ numeric ที่ผู้ใช้ "ไม่ได้ส่งมา"
    num_keys = [k for k, v in defaults.items() if isinstance(v, (int, float, np.integer, np.floating))]
    factors = [1.0 - pct_window, 1.0, 1.0 + pct_window]

    cards: List[Dict[str, Any]] = []
    for f in factors:
        feat = base.copy()
        for k in num_keys:
            if k in fixed:
                continue
            v = defaults.get(k, None)
            if v is not None and isinstance(v, (int, float, np.integer, np.floating)):
                feat[k] = float(v) * f

        cast_int_like(feat)

        # เติมคีย์ให้ครบ กันพลาด
        for k in defaults:
            if k not in feat:
                feat[k] = defaults[k]

        X = pd.DataFrame([feat])
        y_raw = float(PREDICTOR.predict(X)[0])     # อาจเป็น log
        price_usd = invert_target(y_raw)           # แปลงเป็นราคา USD
        price_thb = round(price_usd * float(usd_to_thb), 2)

        cards.append({
            "features": feat,
            "predicted_price_raw": round(y_raw, 4),
            "predicted_price_usd": round(price_usd, 2),
            "predicted_price_thb": price_thb
        })

    return cards

# =========================
# FastAPI app
# =========================
app = FastAPI(title="House Price Cards API", version="1.1.0")

# เปิด CORS สำหรับ dev (โปรดปรับ origin ใน production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas
class ThreeCardsRequest(BaseModel):
    fixed: Dict[str, Any] = Field(default_factory=dict, description="ค่าที่ล็อค เช่น {'LotArea': 9000}")
    pct_window: float = Field(0.15, ge=0.0, le=1.0, description="± สัดส่วนสร้าง low/med/high")
    usd_to_thb: float = Field(36.0, gt=0, description="เรตแปลง USD→THB")

class Card(BaseModel):
    features: Dict[str, Any]
    predicted_price_raw: float
    predicted_price_usd: float
    predicted_price_thb: float

class ThreeCardsResponse(BaseModel):
    count: int
    cards: List[Card]

# Health check
@app.get("/health")
def health():
    return {"status": "ok", "model_file": MODEL_FILE}

# Main endpoint
@app.post("/api/three-cards", response_model=ThreeCardsResponse)
def api_three_cards(payload: ThreeCardsRequest):
    try:
        cards = make_three_cards(
            fixed=payload.fixed,
            pct_window=payload.pct_window,
            usd_to_thb=payload.usd_to_thb,
        )
        return ThreeCardsResponse(count=len(cards), cards=cards)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to build cards: {e}")

# รันแบบ python app.py ก็ได้
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", reload=True, host="0.0.0.0", port=5000)
