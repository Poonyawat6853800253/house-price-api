from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from pydantic import BaseModel
import pandas as pd
import joblib
import json, os

# ========================
# CONFIG
# ========================
SECRET_KEY = "supersecretkey"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

app = FastAPI(title="🏠 House Price Prediction API", version="1.0")

# ========================
# ✅ ENABLE CORS (สำคัญ)
# ========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ถ้าจะ deploy จริง ใส่ domain frontend เช่น "https://your-frontend.vercel.app"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# USER STORAGE (JSON-based)
# ========================
def load_users():
    """โหลด users.json (รองรับทั้ง UTF-8 และ UTF-8-SIG)"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8-sig") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_users(users):
    """บันทึก users.json"""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

fake_users_db = load_users()

# ========================
# MODEL LOADING
# ========================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "house_price_pipeline.joblib")
TRAIN_PATH = os.path.join(BASE_DIR, "train_cleaned.csv")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ ไม่พบไฟล์โมเดลที่ {MODEL_PATH}")
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"❌ ไม่พบไฟล์ train_cleaned.csv ที่ {TRAIN_PATH}")

model = joblib.load(MODEL_PATH)
train_cols = pd.read_csv(TRAIN_PATH).drop(columns=["SalePrice"]).columns

# ========================
# JWT FUNCTIONS
# ========================
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """สร้าง JWT Token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ========================
# AUTH FUNCTIONS
# ========================
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(username: str):
    return fake_users_db.get(username)

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

# ========================
# ROUTES
# ========================
@app.get("/")
def root():
    return {"msg": "✅ House Price Prediction API is running."}

# ---------- Register ----------
@app.post("/register")
def register(username: str, password: str):
    """สมัครสมาชิก"""
    if username in fake_users_db:
        raise HTTPException(status_code=400, detail="User already exists")

    if len(password.encode('utf-8')) > 72:
        raise HTTPException(status_code=400, detail="Password too long (max 72 bytes for bcrypt)")

    hashed_pw = pwd_context.hash(password)
    fake_users_db[username] = {"username": username, "hashed_password": hashed_pw}
    save_users(fake_users_db)
    return {"msg": f"User '{username}' created successfully"}

# ---------- Login ----------
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """เข้าสู่ระบบ"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

# ---------- Predict ----------
class HouseInput(BaseModel):
    OverallQual: int
    GrLivArea: int
    GarageCars: int
    YearBuilt: int
    YearRemodAdd: int
    FullBath: int
    KitchenQual: str
    Neighborhood: str
    LotArea: int
    TotalBsmtSF: int
    FirstFlrSF: int
    SecondFlrSF: int
    Fireplaces: int
    MSZoning: str
    BldgType: str

@app.post("/predict")
def predict(data: HouseInput, token: str = Depends(oauth2_scheme)):
    """ทำนายราคาบ้าน"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in fake_users_db:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    df = pd.DataFrame([data.dict()])
    for col in train_cols:
        if col not in df.columns:
            df[col] = None
    df = df[train_cols]

    pred = model.predict(df)[0]
    return {"predicted_price": round(float(pred), 2)}
