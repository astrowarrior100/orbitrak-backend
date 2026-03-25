import os
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
from engine import find_debris
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
import requests


print("LOADED FILE:", os.path.abspath(__file__))


app = FastAPI()

origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://astrowarrior100.github.io",
    "https://astrowarrior100.github.io/orbitrak-frontend"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # allow all origins (for development)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.get("/test-cors")
def test():
    return {"message": "CORS working"}


class ScanRequest(BaseModel):
    latitude: float
    longitude: float
    radius_km: float

@app.post("/scan")
def scan_debris(request: ScanRequest):
    # Your engine requires min_alt_km and max_alt_km
    min_alt_km = 100
    max_alt_km = 50000

    results = find_debris(
        request.latitude,
        request.longitude,
        min_alt_km,
        max_alt_km,
        request.radius_km
    )

    return {"results": results}


@app.get("/tle/{norad_id}")
def get_tle(norad_id: int):
    # Example: pulling from Celestrak
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
    r = requests.get(url)

    if r.status_code != 200 or "1 " not in r.text:
        raise HTTPException(status_code=404, detail="TLE not found")

    lines = r.text.strip().split("\n")
    return {
        "name": lines[0].strip(),
        "line1": lines[1].strip(),
        "line2": lines[2].strip()
    }
