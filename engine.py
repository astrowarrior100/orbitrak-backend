import os
import requests
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime, timezone, timedelta
import json
import math

from skyfield.api import EarthSatellite, load, wgs84
from geopy.distance import geodesic
import xgboost as xgb

# =============================
# CONFIG
# =============================

SPACE_TRACK_USER = os.getenv("o_USERNAME")
SPACE_TRACK_PASS = os.getenv("o_PASSWORD")

print("ENGINE IMPORTED")
print("USER:", SPACE_TRACK_USER)
print("PASS:", SPACE_TRACK_PASS)


if SPACE_TRACK_USER is None or SPACE_TRACK_PASS is None:
    print("WARNING: Space-Track credentials missing. Running in offline mode.")


LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
GP_PAYLOAD_URL = "https://www.space-track.org/basicspacedata/query/class/gp/OBJECT_TYPE/PAYLOAD/decay_date/null-val/epoch/>now-30/format/json"
GP_RB_URL = "https://www.space-track.org/basicspacedata/query/class/gp/OBJECT_TYPE/ROCKET BODY/decay_date/null-val/epoch/>now-30/format/json"
GP_DEBRIS_URL = "https://www.space-track.org/basicspacedata/query/class/gp/OBJECT_TYPE/DEBRIS/decay_date/null-val/epoch/>now-30/format/json"
SATCAT_URL = "https://www.space-track.org/basicspacedata/query/class/satcat/format/json"
SATCAT_CACHE_PATH = "satcat_cache.json"
GP_CACHE_PATH = "gp_cache.json"
GP_CACHE_MAX_AGE_MIN = 360  # 6 hours


SIZE_MODEL_PATH = "size_xgb.json"
MASS_MODEL_PATH = "mass_xgb.json"

SESSION = None
GP_DATA = None
SATCAT_DATA = None

def get_session():
    global SESSION
    if SESSION is None:
        SESSION = requests.Session()
        SESSION.post(LOGIN_URL, data={"identity": SPACE_TRACK_USER, "password": SPACE_TRACK_PASS})
    return SESSION


#def load_processed_cache():
    if not os.path.exists(PROCESSED_CACHE_FILE):
        return None

    with open(PROCESSED_CACHE_FILE, "r") as f:
        data = json.load(f)

    # Check expiry
    timestamp = datetime.fromisoformat(data.get("timestamp"))
    if datetime.utcnow() - timestamp > timedelta(days=CACHE_EXPIRY_DAYS):
        return None

    return data.get("objects", [])


#def save_processed_cache(objects):
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "objects": objects
    }
    with open(PROCESSED_CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)

# =============================
# SATCAT CACHING
# =============================

def _load_satcat_cache():
    if not os.path.exists(SATCAT_CACHE_PATH):
        return None
    try:
        with open(SATCAT_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_satcat_cache(data):
    try:
        with open(SATCAT_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "data": data
            }, f)
    except Exception:
        pass

def _satcat_cache_is_fresh(cache):
    if not cache or "fetched_at" not in cache:
        return False
    try:
        t = datetime.fromisoformat(cache["fetched_at"])
    except Exception:
        return False
    return (datetime.now(timezone.utc) - t) < timedelta(hours=24)

def _load_gp_cache():
    if not os.path.exists(GP_CACHE_PATH):
        return None
    try:
        with open(GP_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_gp_cache(data):
    try:
        with open(GP_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "data": data
            }, f)
    except Exception:
        pass

def _gp_cache_is_fresh(cache):
    if not cache or "fetched_at" not in cache:
        return False
    try:
        t = datetime.fromisoformat(cache["fetched_at"])
    except Exception:
        return False
    return (datetime.now(timezone.utc) - t) < timedelta(minutes=GP_CACHE_MAX_AGE_MIN)

def fetch_all_gp_objects():
    global GP_DATA

    if GP_DATA is not None:
        return GP_DATA

    cache = _load_gp_cache()
    if not _gp_cache_is_fresh(cache):
        payload = fetch_json(GP_PAYLOAD_URL)
        rb      = fetch_json(GP_RB_URL)
        debris  = fetch_json(GP_DEBRIS_URL)

        data = {
            "payload": payload,
            "rocket_body": rb,
            "debris": debris
        }

        _save_gp_cache(data)
        GP_DATA = data
    else:
        GP_DATA = cache["data"]

    return GP_DATA


# =============================
# BASIC HELPERS
# =============================

def fetch_json(url):
    session = get_session()
    resp = session.get(url)
    resp.raise_for_status()
    return resp.json()


def build_satcat_minimal():
    global SATCAT_DATA

    if SATCAT_DATA is not None:
        return SATCAT_DATA

    cache = _load_satcat_cache()
    if not _satcat_cache_is_fresh(cache):
        satcat_json = fetch_json(SATCAT_URL)
        _save_satcat_cache(satcat_json)
    else:
        satcat_json = cache["data"]

    satcat_lookup = {}

    for obj in satcat_json:
        if not isinstance(obj, dict):
            continue

        norad = obj.get("NORAD_CAT_ID")
        if not norad:
            continue

        rcs = obj.get("RCS_SIZE") or "Unknown"
        country = obj.get("COUNTRY") or "Unknown"

        satcat_lookup[norad] = {
            "rcs_size": rcs,
            "country": country
        }

    SATCAT_DATA = satcat_lookup
    return SATCAT_DATA


def to_utc(dt_str):
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def build_satellites(json_data):
    satellites = []

    for obj in json_data:
        if not isinstance(obj, dict):
            continue

        name = obj.get("OBJECT_NAME")
        norad = obj.get("NORAD_CAT_ID")
        line1 = obj.get("TLE_LINE1")
        line2 = obj.get("TLE_LINE2")
        epoch_str = obj.get("EPOCH")
        object_id = obj.get("OBJECT_ID")
        object_type = obj.get("OBJECT_TYPE")

        if not (name and norad and line1 and line2):
            continue

        tle_epoch = to_utc(epoch_str)
        if tle_epoch is None:
            continue

        # Skip TLEs older than 30 days (Skyfield cannot propagate them reliably)
        if (datetime.now(timezone.utc) - tle_epoch).days > 30:
            continue


        sat = EarthSatellite(line1, line2, name)

        satellites.append({
            "sat": sat,
            "name": name,
            "norad_id": norad,
            "object_id": object_id,
            "object_type": object_type,
            "tle_epoch": tle_epoch
        })

    return satellites


def compute_orbital_parameters(sat, ts):
    t = ts.now()

    geocentric = sat.at(t)
    vel = geocentric.velocity.km_per_s
    vel_mag = np.linalg.norm(vel)

    elements = sat.model

    ecc = elements.ecco
    inc = np.degrees(elements.inclo)
    raan = np.degrees(elements.nodeo)
    argp = np.degrees(elements.argpo)
    mean_anom = np.degrees(elements.mo)

    sma_km = elements.a * 6378.137

    period = elements.no_kozai
    if period != 0:
        period = (2 * np.pi) / period / 60
    else:
        period = None

    perigee = sma_km * (1 - ecc) - 6378.137
    apogee = sma_km * (1 + ecc) - 6378.137

    return {
        "velocity_kms": vel_mag,
        "eccentricity": ecc,
        "inclination_deg": inc,
        "raan_deg": raan,
        "arg_perigee_deg": argp,
        "mean_anomaly_deg": mean_anom,
        "semi_major_axis_km": sma_km,
        "period_min": period,
        "perigee_km": perigee,
        "apogee_km": apogee
    }
# =============================
# SIZE ENGINE (HYBRID PHYSICS + ML)
# =============================

@dataclass
class SizeFeatures:
    rcs: Optional[str]
    object_type: Optional[str]
    material: Optional[str]
    orbit_zone: int
    ecc: float
    inc_deg: float
    perigee_km: float
    apogee_km: float
    period_min: float


class SizeEngineV3:
    def __init__(self, model_path: str = SIZE_MODEL_PATH, debug: bool = False):
        self.model = None
        self.debug = debug

        if os.path.exists(model_path):
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
        else:
            print(f"SizeEngineV3 WARNING: No ML model at {model_path}. Using physics-only size estimates.")

    def _encode_rcs(self, rcs_size: Optional[str]) -> int:
        if not rcs_size:
            return 0
        r = str(rcs_size).strip().upper()
        if r == "SMALL":
            return 1
        if r == "MEDIUM":
            return 2
        if r == "LARGE":
            return 3
        return 0

    def _encode_object_type(self, object_type: Optional[str]) -> int:
        if not object_type:
            return 0
        t = str(object_type).strip().upper()
        if t == "PAYLOAD":
            return 1
        if t == "ROCKET BODY":
            return 2
        if t == "DEBRIS":
            return 3
        return 0

    def _encode_material(self, material: Optional[str]) -> int:
        if not material:
            return 0
        m = str(material).lower()
        if "steel" in m:
            return 3
        if "titanium" in m:
            return 4
        if "composite" in m or "carbon fiber" in m:
            return 2
        if "aluminum" in m or "aluminium" in m:
            return 1
        return 5  # mixed/unknown

    def _safe(self, v, default=0.0):
        try:
            if v is None:
                return default
            return float(v)
        except (TypeError, ValueError):
            return default

    def _physics_size_cm(self, rcs_code: int, obj_code: int) -> float:
        if rcs_code == 1:
            base = 5.0
        elif rcs_code == 2:
            base = 30.0
        elif rcs_code == 3:
            base = 100.0
        else:
            base = 10.0

        if obj_code == 1:
            base *= 1.5
        elif obj_code == 2:
            base *= 2.0
        elif obj_code == 3:
            base *= 0.8

        return base

    def _orbit_adjustment_factor(self, ecc: float, inc_deg: float, perigee_km: float, apogee_km: float) -> float:
        f = 1.0
        if ecc > 0.2:
            f *= 1.2
        if ecc > 0.5:
            f *= 1.4
        if perigee_km < 300:
            f *= 0.8
        if 60 <= inc_deg <= 70:
            f *= 1.05
        return f

    def estimate_size(self, feats: SizeFeatures):
        rcs_code = self._encode_rcs(feats.rcs)
        obj_code = self._encode_object_type(feats.object_type)
        mat_code = self._encode_material(feats.material)

        ecc = max(0.0, min(self._safe(feats.ecc), 0.99))
        inc = max(0.0, min(self._safe(feats.inc_deg), 180.0))
        perigee = self._safe(feats.perigee_km)
        apogee = max(perigee, self._safe(feats.apogee_km))
        period = max(0.0, self._safe(feats.period_min))
        orbit_zone = int(feats.orbit_zone)

        base_size = self._physics_size_cm(rcs_code, obj_code)
        orbit_factor = self._orbit_adjustment_factor(ecc, inc, perigee, apogee)
        physics_size = base_size * orbit_factor

        ml_size = None
        factor = 1.0

        if self.model is not None:
            X = np.array([[
                rcs_code,
                obj_code,
                mat_code,
                orbit_zone,
                ecc,
                inc,
                perigee,
                apogee,
                period
            ]], dtype=float)

            if self.debug:
                print("\n[DEBUG SIZE] Features:", feats)
                print("[DEBUG SIZE] X vector:", X)

            raw_pred = float(self.model.predict(X)[0])
            ml_size = max(raw_pred, 0.0)

            if self.debug:
                print("[DEBUG SIZE] Physics size (cm):", physics_size)
                print("[DEBUG SIZE] ML size (cm):", ml_size)

            if physics_size > 0:
                factor = ml_size / physics_size
            else:
                factor = 1.0

            factor = max(0.5, min(factor, 2.0))

        if self.model is None:
            final_size = physics_size
            note = "Physics-only size estimate (no ML model loaded)"
        else:
            corrected = physics_size * factor
            final_size = 0.7 * physics_size + 0.3 * corrected
            note = "Hybrid physics + ML size estimate"

        min_size = max(final_size * 0.7, 1.0)
        max_size = final_size * 1.3

        if self.debug:
            print("[DEBUG SIZE] Final size (cm):", final_size)
            print("[DEBUG SIZE] Size range (cm):", (min_size, max_size))
            print("[DEBUG SIZE] Factor:", factor)
            print("[DEBUG SIZE] Note:", note)

        return {
            "char_size_cm": final_size,
            "approx_min_size_cm": min_size,
            "approx_max_size_cm": max_size,
            "physics_size_cm": physics_size,
            "ml_size_cm": ml_size,
            "factor": factor,
            "note": note
        }


# =============================
# MASS ENGINE (HYBRID PHYSICS + ML)
# =============================

@dataclass
class MassFeatures:
    rcs: Optional[str]
    object_type: Optional[str]
    material: Optional[str]
    orbit_zone: int
    ecc: float
    inc_deg: float
    perigee_km: float
    apogee_km: float
    period_min: float
    min_size_cm: float
    max_size_cm: float


class MassEngineV3:
    def __init__(self, model_path: str = MASS_MODEL_PATH, debug: bool = False):
        self.model = None
        self.debug = debug

        if os.path.exists(model_path):
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
        else:
            print(f"MassEngineV3 WARNING: No ML model at {model_path}. Using physics-only mass estimates.")

    def _encode_rcs(self, rcs_size: Optional[str]) -> int:
        if not rcs_size:
            return 0
        r = str(rcs_size).strip().upper()
        if r == "SMALL":
            return 1
        if r == "MEDIUM":
            return 2
        if r == "LARGE":
            return 3
        return 0

    def _encode_object_type(self, object_type: Optional[str]) -> int:
        if not object_type:
            return 0
        t = str(object_type).strip().upper()
        if t == "PAYLOAD":
            return 1
        if t == "ROCKET BODY":
            return 2
        if t == "DEBRIS":
            return 3
        return 0

    def _encode_material(self, material: Optional[str]) -> int:
        if not material:
            return 0
        m = str(material).lower()
        if "steel" in m:
            return 3
        if "titanium" in m:
            return 4
        if "composite" in m or "carbon fiber" in m:
            return 2
        if "aluminum" in m or "aluminium" in m:
            return 1
        return 5

    def _safe(self, v, default=0.0):
        try:
            if v is None:
                return default
            return float(v)
        except (TypeError, ValueError):
            return default

    def _material_density(self, material_code: int) -> float:
        if material_code == 1:
            return 1.0
        if material_code == 2:
            return 0.7
        if material_code == 3:
            return 2.0
        if material_code == 4:
            return 1.5
        return 1.0

    def _type_factor(self, obj_code: int) -> float:
        if obj_code == 1:
            return 2.0
        if obj_code == 2:
            return 3.0
        if obj_code == 3:
            return 1.0
        return 1.0

    def _physics_mass(self, min_size_cm: float, max_size_cm: float,
                      material_code: int, obj_code: int) -> float:
        size = max(min_size_cm, max_size_cm)
        if size <= 0:
            return 0.0
        volume_unit = (size ** 3) * 1e-3
        density = self._material_density(material_code)
        type_factor = self._type_factor(obj_code)
        mass = volume_unit * density * type_factor
        return max(mass, 0.0)

    def estimate_mass(self, feats: MassFeatures):
        rcs_code = self._encode_rcs(feats.rcs)
        obj_code = self._encode_object_type(feats.object_type)
        mat_code = self._encode_material(feats.material)

        ecc = max(0.0, min(self._safe(feats.ecc), 0.99))
        inc = max(0.0, min(self._safe(feats.inc_deg), 180.0))
        perigee = self._safe(feats.perigee_km)
        apogee = max(perigee, self._safe(feats.apogee_km))
        period = max(0.0, self._safe(feats.period_min))
        orbit_zone = int(feats.orbit_zone)

        min_size = max(0.0, self._safe(feats.min_size_cm))
        max_size = max(min_size, self._safe(feats.max_size_cm))

        physics_mass = self._physics_mass(min_size, max_size, mat_code, obj_code)

        ml_mass = None
        factor = 1.0

        if self.model is not None:
            X = np.array([[
                rcs_code,
                obj_code,
                mat_code,
                orbit_zone,
                ecc,
                inc,
                perigee,
                apogee,
                period,
                min_size,
                max_size
            ]], dtype=float)

            if self.debug:
                print("\n[DEBUG MASS] Features:", feats)
                print("[DEBUG MASS] X vector:", X)
                print("[DEBUG MASS] Physics mass (kg):", physics_mass)

            raw_pred = float(self.model.predict(X)[0])
            ml_mass = max(raw_pred, 0.0)

            if self.debug:
                print("[DEBUG MASS] ML mass (kg):", ml_mass)

            if physics_mass > 0:
                factor = ml_mass / physics_mass
            else:
                factor = 1.0

            factor = max(0.3, min(factor, 3.0))

        if self.model is None:
            final_mass = physics_mass
            note = "Physics-only mass estimate (no ML model loaded)"
        else:
            corrected = physics_mass * factor
            final_mass = 0.7 * physics_mass + 0.3 * corrected
            final_mass = max(final_mass, 0.0)
            note = "Hybrid physics + ML mass estimate"

        if self.debug:
            print("[DEBUG MASS] Final mass (kg):", final_mass)
            print("[DEBUG MASS] Factor:", factor)
            print("[DEBUG MASS] Note:", note)

        return {
            "mass_kg": final_mass,
            "physics_mass_kg": physics_mass,
            "ml_mass_kg": ml_mass,
            "ml_factor": factor,
            "note": note
        }


# =============================
# BALLISTIC COEFFICIENT
# =============================

def compute_cross_sectional_area(size_cm: float) -> float:
    """Convert characteristic size (diameter in cm) into cross-sectional area (m^2)."""
    radius_m = (size_cm / 100.0) / 2.0
    return math.pi * radius_m * radius_m

def get_drag_coefficient(object_type: str) -> float:
    """Assign a drag coefficient based on object type."""
    if not object_type:
        return 2.2
    t = object_type.lower()
    if "rocket" in t:
        return 2.0
    if "payload" in t:
        return 2.2
    if "debris" in t:
        return 2.3
    return 2.2

def compute_ballistic_coefficient(size_cm: float, mass_kg: float, object_type: str) -> float:
    """Compute ballistic coefficient BC = m / (Cd * A)."""
    A = compute_cross_sectional_area(size_cm)
    Cd = get_drag_coefficient(object_type)
    if A <= 0 or Cd <= 0:
        return 0.0
    return mass_kg / (Cd * A)


# =============================
# DANGER ML ENGINE
# =============================

@dataclass
class DangerFeatures:
    size_min_cm: float
    size_max_cm: float
    mass_kg: float
    velocity_kms: float
    eccentricity: float
    inclination_deg: float
    perigee_km: float
    apogee_km: float
    orbit_zone: int
    collision_risk: float


class DangerEngine:
    def __init__(self, model_path="danger_xgb.json"):
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        
    def estimate_danger(self, feats: DangerFeatures) -> float:
        X = np.array([[
            feats.size_min_cm,
            feats.size_max_cm,
            feats.mass_kg,
            feats.velocity_kms,
            feats.eccentricity,
            feats.inclination_deg,
            feats.perigee_km,
            feats.apogee_km,
            feats.orbit_zone,
            feats.collision_risk
        ]], dtype=float)

        pred = float(self.model.predict(X)[0])
        return max(0.0, min(pred, 10.0))  # keep in 0–10 range


danger_engine = DangerEngine()
# =============================
# MATERIAL ENGINE (ENHANCED)
# =============================

@dataclass
class DebrisMaterialContext:
    object_name: str
    parent_object_types: List[str]
    rcs_size: Optional[str] = None
    country: Optional[str] = None


def _guess_from_name(name_upper: str):
    if "R/B" in name_upper or "RKT" in name_upper or "STG" in name_upper:
        return "Aluminum alloy + steel (rocket structure)", 0.6
    if "PANEL" in name_upper or "SOLAR" in name_upper:
        return "Silicon + glass on composite substrate", 0.5
    if "FAIRING" in name_upper:
        return "Carbon fiber composite", 0.6
    if "TANK" in name_upper:
        return "Titanium or aluminum pressure vessel", 0.6
    if "ANT" in name_upper:
        return "Aluminum/composite antenna structure", 0.4
    if "ADAPTER" in name_upper:
        return "Aluminum or composite adapter", 0.4
    return None, 0.0


def _guess_from_parent_types(parent_types):
    types_upper = [t.upper() for t in parent_types]

    if "ROCKET BODY" in types_upper:
        return "Aluminum alloy + steel (rocket body debris)", 0.6
    if "PAYLOAD" in types_upper:
        return "Aluminum + composite structures (satellite debris)", 0.5
    if "ROCKET BODY" in types_upper and "PAYLOAD" in types_upper:
        return "Mixed satellite + rocket materials", 0.5
    return None, 0.0


def _guess_from_rcs(rcs_size: Optional[str]):
    if not rcs_size:
        return None, 0.0
    r = rcs_size.upper()
    if r == "LARGE":
        return "Large metallic structure (likely aluminum/steel)", 0.4
    if r == "MEDIUM":
        return "Medium metallic/composite structure", 0.3
    if r == "SMALL":
        return "Small composite or aluminum fragment", 0.2
    return None, 0.0


def _guess_from_country(country: Optional[str]):
    if not country:
        return None, 0.0
    c = country.upper()

    if c in ["US", "USA"]:
        return "Aluminum + composite (typical US satellite materials)", 0.3
    if c in ["RU", "RUS", "CIS"]:
        return "Aluminum + steel (common Russian launch materials)", 0.3
    if c in ["CN", "CHN"]:
        return "Aluminum alloy + composite (Chinese spacecraft)", 0.3
    if c in ["IN", "IND"]:
        return "Aluminum + composite (Indian spacecraft)", 0.3
    if c in ["EU", "ESA"]:
        return "Carbon composite + aluminum (ESA spacecraft)", 0.3

    return None, 0.1


def estimate_debris_material(ctx: DebrisMaterialContext):
    name_guess, name_conf = _guess_from_name(ctx.object_name.upper())
    parent_guess, parent_conf = _guess_from_parent_types(ctx.parent_object_types)
    rcs_guess, rcs_conf = _guess_from_rcs(ctx.rcs_size)
    country_guess, country_conf = _guess_from_country(ctx.country)

    guesses = []
    if name_guess: guesses.append((name_guess, name_conf))
    if parent_guess: guesses.append((parent_guess, parent_conf))
    if rcs_guess: guesses.append((rcs_guess, rcs_conf))
    if country_guess: guesses.append((country_guess, country_conf))

    if not guesses:
        return {"material": "Unknown", "confidence": "LOW"}

    total_conf = sum(conf for _, conf in guesses)
    avg_conf = total_conf / len(guesses)

    best_guess = max(guesses, key=lambda x: x[1])[0]

    if avg_conf >= 0.55:
        conf_label = "HIGH"
    elif avg_conf >= 0.35:
        conf_label = "MEDIUM"
    else:
        conf_label = "LOW"

    return {
        "material": best_guess,
        "confidence": conf_label,
        "sources_used": {
            "name": bool(name_guess),
            "parent_type": bool(parent_guess),
            "rcs": bool(rcs_guess),
            "country": bool(country_guess)
        }
    }


# =============================
# ORBIT CLASSIFICATION
# =============================

def classify_orbit_v2(alt_km, inc_deg, ecc, perigee_km, apogee_km, period_min, argp_deg):
    if alt_km < 160:
        return {"orbit": "Atmospheric / Decaying", "notes": "Below stable orbit altitude."}

    if alt_km < 2000:
        zone = "LEO"
    elif alt_km < 35786:
        zone = "MEO"
    elif abs(alt_km - 35786) <= 200:
        zone = "GEO"
    else:
        zone = "HEO"

    if 600 <= alt_km <= 900 and 96.5 <= inc_deg <= 99.5:
        return {"orbit": "Sun-Synchronous Orbit (SSO)", "notes": "Inclination matches sun-synchronous regression band."}

    if 85 <= inc_deg <= 95:
        return {"orbit": f"{zone} Polar Orbit", "notes": "Inclination near 90°."}

    if inc_deg < 5:
        return {"orbit": f"{zone} Equatorial Orbit", "notes": "Inclination near 0°."}

    if zone == "GEO":
        if ecc > 0.01 or abs(inc_deg) > 1:
            return {"orbit": "Inclined GEO / Drifting GEO", "notes": "Object is near GEO altitude but drifting."}
        return {"orbit": "Geostationary Orbit (GEO)", "notes": "Near 35,786 km altitude."}

    if 36000 <= alt_km <= 37000 and ecc < 0.01:
        return {"orbit": "GEO Graveyard Orbit", "notes": "Supersynchronous disposal orbit."}

    if ecc > 0.1 and perigee_km < 2000 and 33000 <= apogee_km <= 40000:
        return {"orbit": "GTO (Geostationary Transfer Orbit)", "notes": "High eccentricity with GEO apogee."}

    if ecc > 0.5 and 62 <= inc_deg <= 65 and 690 <= period_min <= 750:
        return {"orbit": "Molniya Orbit", "notes": "Highly elliptical 12-hour orbit."}

    if ecc > 0.2 and 62 <= inc_deg <= 65 and 1380 <= period_min <= 1500:
        return {"orbit": "Tundra Orbit", "notes": "24-hour highly elliptical orbit."}

    if ecc > 0.2 and zone == "HEO":
        return {"orbit": "Highly Elliptical Orbit (HEO)", "notes": "High eccentricity beyond GEO altitude."}

    return {"orbit": f"{zone} Orbit", "notes": "Standard orbital classification."}


def get_orbit_zone(alt_km):
    if alt_km < 2000:
        return 1
    if alt_km < 35786:
        return 2
    if abs(alt_km - 35786) <= 200:
        return 3
    return 4


# =============================
# RISK HELPERS
# =============================

def _norm(value, vmin, vmax):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= vmin:
        return 0.0
    if v >= vmax:
        return 1.0
    return (v - vmin) / (vmax - vmin)


def _material_risk(material_str: str) -> Optional[float]:
    if not material_str:
        return None
    m = material_str.lower()
    if "steel" in m:
        return 1.0
    if "titanium" in m:
        return 0.9
    if "tank" in m:
        return 0.9
    if "carbon" in m or "composite" in m:
        return 0.6
    if "aluminum" in m or "aluminium" in m:
        return 0.7
    if "silicon" in m or "glass" in m:
        return 0.5
    return 0.7


def _rcs_code(rcs_size: str) -> Optional[float]:
    if not rcs_size:
        return None
    r = rcs_size.strip().upper()
    if r == "SMALL":
        return 0.3
    if r == "MEDIUM":
        return 0.6
    if r == "LARGE":
        return 1.0
    return None


# =============================
# COLLISION RISK (0–100)
# =============================

def compute_collision_risk(obj) -> float:
    size_info = obj.get("size_estimate", {})
    material_info = obj.get("material_estimate", {})

    try:
        min_size_cm = float(size_info.get("approx_min_size_cm", 0.0))
        max_size_cm = float(size_info.get("approx_max_size_cm", 0.0))
    except (TypeError, ValueError):
        min_size_cm = max_size_cm = 0.0

    char_size_cm = max(min_size_cm, max_size_cm)

    v_kms = obj.get("velocity_kms")
    ecc = obj.get("eccentricity")
    inc = obj.get("inclination_deg")
    perigee = obj.get("perigee_km")
    alt = obj.get("altitude_km")
    rcs_size = obj.get("rcs_size")
    material_str = material_info.get("material")

    orbit_zone = get_orbit_zone(alt) if alt is not None else None

    factors = {}
    weights = {}

    s = _norm(char_size_cm, 1.0, 300.0)
    if s is not None:
        factors["size"] = s
        weights["size"] = 1.5

    v = _norm(v_kms, 6.0, 8.0)
    if v is not None:
        factors["velocity"] = v
        weights["velocity"] = 1.0

    if orbit_zone is not None:
        if orbit_zone == 1:
            d = 1.0
        elif orbit_zone == 2:
            d = 0.6
        elif orbit_zone == 3:
            d = 0.4
        else:
            d = 0.3
        factors["density"] = d
        weights["density"] = 1.5

    e = _norm(ecc, 0.0, 0.8)
    if e is not None:
        factors["eccentricity"] = e
        weights["eccentricity"] = 0.7

    if inc is not None:
        if 96.5 <= inc <= 99.5:
            i = 1.0
        elif 85 <= inc <= 95:
            i = 0.9
        else:
            i = _norm(inc, 0.0, 120.0) or 0.5
        factors["inclination"] = i
        weights["inclination"] = 1.0

    p = _norm(perigee, 160.0, 40000.0)
    if p is not None:
        p = 1.0 - p
        factors["perigee"] = p
        weights["perigee"] = 1.0

    r = _rcs_code(rcs_size)
    if r is not None:
        factors["rcs"] = r
        weights["rcs"] = 0.8

    m = _material_risk(material_str)
    if m is not None:
        factors["material"] = m
        weights["material"] = 0.3

    if not factors:
        return 0.0

    num = 0.0
    den = 0.0
    for k, f in factors.items():
        w = weights.get(k, 1.0)
        num += w * f
        den += w

    score_0_1 = num / den
    return max(0.0, min(100.0, score_0_1 * 100.0))


# =============================
# DANGER SCORE (0–10)
# =============================

def compute_danger_score(obj, collision_risk: Optional[float] = None) -> float:
    size_info = obj.get("size_estimate", {})
    mass_info = obj.get("mass_estimate", {})
    material_info = obj.get("material_estimate", {})

    try:
        min_size_cm = float(size_info.get("approx_min_size_cm", 0.0))
        max_size_cm = float(size_info.get("approx_max_size_cm", 0.0))
    except (TypeError, ValueError):
        min_size_cm = max_size_cm = 0.0

    char_size_cm = max(min_size_cm, max_size_cm)
    mass_kg = mass_info.get("mass_kg")
    material_str = material_info.get("material")

    ecc = obj.get("eccentricity")
    perigee = obj.get("perigee_km")
    alt = obj.get("altitude_km")
    orbit_zone = get_orbit_zone(alt) if alt is not None else None

    factors = {}
    weights = {}

    if collision_risk is None:
        collision_risk = compute_collision_risk(obj)
    coll_norm = collision_risk / 100.0
    factors["collision"] = coll_norm
    weights["collision"] = 2.0

    s = _norm(char_size_cm, 1.0, 300.0)
    if s is not None:
        factors["size_severity"] = s
        weights["size_severity"] = 1.0

    m_norm = _norm(mass_kg, 0.1, 5000.0)
    if m_norm is not None:
        factors["mass_severity"] = m_norm
        weights["mass_severity"] = 1.5

    mat_risk = _material_risk(material_str)
    if mat_risk is not None:
        factors["material_severity"] = mat_risk
        weights["material_severity"] = 0.7

    if orbit_zone is not None:
        if orbit_zone == 1:
            oh = 1.0
        elif orbit_zone == 2:
            oh = 0.7
        elif orbit_zone == 3:
            oh = 0.5
        else:
            oh = 0.4
        factors["orbit_zone_hazard"] = oh
        weights["orbit_zone_hazard"] = 1.0

    p = _norm(perigee, 160.0, 40000.0)
    if p is not None:
        p = 1.0 - p
        factors["perigee_hazard"] = p
        weights["perigee_hazard"] = 1.2

    e = _norm(ecc, 0.0, 0.8)
    if e is not None:
        factors["ecc_hazard"] = e
        weights["ecc_hazard"] = 0.6

    if not factors:
        return 0.0

    num = 0.0
    den = 0.0
    for k, f in factors.items():
        w = weights.get(k, 1.0)
        num += w * f
        den += w

    score_0_1 = num / den
    return max(0.0, min(10.0, score_0_1 * 10.0))
# =============================
# PHYSICS-BASED DECAY RATE ENGINE
# =============================

@dataclass
class DecayFeatures:
    ballistic_coefficient: float  # kg/m^2
    mean_altitude_km: float       # km
    perigee_km: float             # km
    apogee_km: float              # km


class DecayRateEnginePhysics:
    """
    Simple physics-based orbital decay rate estimator.
    Output: approximate decay rate in km/day (positive number = km lost per day).
    """

    def __init__(self):
        self.scale = 1.0  # tuning constant

    def _atmospheric_density(self, h_km: float) -> float:
        """Exponential atmosphere model (very rough)."""
        if h_km < 120:
            h_km = 120.0
        H = 50.0          # scale height (km)
        rho0 = 5e-7       # density at 200 km (kg/m^3)
        return rho0 * math.exp(-(h_km - 200.0) / H)

    def _orbital_velocity(self, h_km: float) -> float:
        """Circular orbit velocity at altitude h_km (km/s)."""
        mu = 398600.4418  # km^3/s^2
        Re = 6378.137     # km
        r = Re + h_km
        return math.sqrt(mu / r)

    def estimate_decay_rate(self, feats: DecayFeatures) -> float:
        bc = max(feats.ballistic_coefficient, 1e-3)
        h = float(feats.mean_altitude_km)

        rho = self._atmospheric_density(h)
        v_kms = self._orbital_velocity(h)
        v_ms = v_kms * 1000.0

        a_drag = 0.5 * rho * (v_ms ** 2) / bc  # m/s^2

        dh_dt_km_per_s = -a_drag / 9.81 * 0.1
        dh_dt_km_per_day = dh_dt_km_per_s * 86400.0

        return abs(dh_dt_km_per_day) * self.scale


# =============================
# INIT ENGINES (DEBUG ON)
# =============================

size_engine = SizeEngineV3(debug=False)
mass_engine = MassEngineV3(debug=False)
decay_engine_physics = DecayRateEnginePhysics()


# =============================
# MAIN SEARCH ENGINE
# =============================


    #print("Building processed debris cache...")

    # 1. Load TLE + SATCAT (already cached)
    # 2. Propagate orbits
    # 3. Compute orbital parameters
    # 4. Run ML models
    # 5. Compute danger scores
    # 6. Build final debris objects

    #objects = []  # ← your final debris list

    # Your existing debris-building loop goes here
    # For each debris object, append the full dictionary to `objects`

    #save_processed_cache(objects)
    #print("Processed debris cache saved.")
    #return objects


def find_debris(lat, lon, min_alt_km, max_alt_km, radius_km):
    gp_data = fetch_all_gp_objects()

    gp_payloads = build_satellites(gp_data["payload"])
    gp_rocket_bodies = build_satellites(gp_data["rocket_body"])
    gp_debris = build_satellites(gp_data["debris"])

    all_gp_objects = gp_payloads + gp_rocket_bodies + gp_debris

    satcat_min = build_satcat_minimal()
    ts = load.timescale()

    results = []

    for entry in all_gp_objects:
        sat = entry["sat"]
        name = entry["name"]
        norad = entry["norad_id"]
        object_id = entry["object_id"]
        object_type = entry["object_type"]

        # Only debris
        if object_type != "DEBRIS":
            continue

        geocentric = sat.at(ts.now())
        subpoint = wgs84.subpoint(geocentric)
        sat_lat = subpoint.latitude.degrees
        sat_lon = subpoint.longitude.degrees
        sat_alt = subpoint.elevation.km

        # Skip invalid coordinate cases
        if math.isnan(sat_lat) or math.isnan(sat_lon) or math.isnan(sat_alt):
            continue

        # ⭐ NEW: Altitude filtering
        if sat_alt < min_alt_km:
            continue
        if sat_alt > max_alt_km:
            continue

        # Distance filtering
        dist = geodesic((lat, lon), (sat_lat, sat_lon)).km
        if dist > radius_km:
            continue

        # (Everything below remains unchanged)
        params = compute_orbital_parameters(sat, ts)
        satcat_info = satcat_min.get(norad, {})
        rcs = satcat_info.get("rcs_size", "Unknown")
        country = satcat_info.get("country", "Unknown")

        parent_types = []
        if object_id:
            prefix = object_id[:8]
            for obj in all_gp_objects:
                if obj["object_id"] and obj["object_id"].startswith(prefix) and obj["object_type"] != "DEBRIS":
                    parent_types.append(obj["object_type"])

        material_ctx = DebrisMaterialContext(
            object_name=name,
            parent_object_types=parent_types,
            rcs_size=rcs
        )
        material_info = estimate_debris_material(material_ctx)

        size_feats = SizeFeatures(
            rcs=rcs,
            object_type=object_type,
            material=material_info["material"],
            orbit_zone=get_orbit_zone(sat_alt),
            ecc=params["eccentricity"],
            inc_deg=params["inclination_deg"],
            perigee_km=params["perigee_km"],
            apogee_km=params["apogee_km"],
            period_min=params["period_min"] if params["period_min"] else 0.0
        )
        size_ml = size_engine.estimate_size(size_feats)

        size_info = {
            "rcs_size": rcs,
            "approx_min_size_cm": f"{size_ml['approx_min_size_cm']:.1f}",
            "approx_max_size_cm": f"{size_ml['approx_max_size_cm']:.1f}",
            "note": size_ml["note"]
        }

        orbit_info = classify_orbit_v2(
            alt_km=sat_alt,
            inc_deg=params["inclination_deg"],
            ecc=params["eccentricity"],
            perigee_km=params["perigee_km"],
            apogee_km=params["apogee_km"],
            period_min=params["period_min"],
            argp_deg=params["arg_perigee_deg"]
        )

        min_size_val = float(size_ml["approx_min_size_cm"])
        max_size_val = float(size_ml["approx_max_size_cm"])

        mass_feats = MassFeatures(
            rcs=rcs,
            object_type=object_type,
            material=material_info["material"],
            orbit_zone=get_orbit_zone(sat_alt),
            ecc=params["eccentricity"],
            inc_deg=params["inclination_deg"],
            perigee_km=params["perigee_km"],
            apogee_km=params["apogee_km"],
            period_min=params["period_min"] if params["period_min"] else 0.0,
            min_size_cm=min_size_val,
            max_size_cm=max_size_val
        )
        mass_info = mass_engine.estimate_mass(mass_feats)

        char_size_cm = size_ml["char_size_cm"]
        mass_kg = mass_info.get("mass_kg", 0.0)
        bc = compute_ballistic_coefficient(char_size_cm, mass_kg, object_type)

        perigee_raw = params.get("perigee_km")
        apogee_raw = params.get("apogee_km")

        try:
            perigee = float(perigee_raw)
        except (TypeError, ValueError):
            perigee = float(sat_alt)

        try:
            apogee = float(apogee_raw)
        except (TypeError, ValueError):
            apogee = float(sat_alt)

        mean_alt = 0.5 * (perigee + apogee)

        decay_feats = DecayFeatures(
            ballistic_coefficient=bc,
            mean_altitude_km=mean_alt,
            perigee_km=perigee,
            apogee_km=apogee
        )

        decay_rate = decay_engine_physics.estimate_decay_rate(decay_feats)

        obj = {
            "name": name,
            "norad_id": norad,
            "object_id": object_id,
            "object_type": "DEBRIS",
            "orbit_classification": orbit_info,
            "altitude_km": sat_alt,
            **params,
            "ballistic_coefficient": bc,
            "decay_rate_km_per_day": decay_rate,
            "country": country,
            "rcs_size": rcs,
            "size_estimate": size_info,
            "material_estimate": material_info,
            "mass_estimate": mass_info
        }

        coll_risk = compute_collision_risk(obj)
        analytic_danger = compute_danger_score(obj, coll_risk)

        danger_feats = DangerFeatures(
            size_min_cm=min_size_val,
            size_max_cm=max_size_val,
            mass_kg=mass_info["mass_kg"],
            velocity_kms=params["velocity_kms"],
            eccentricity=params["eccentricity"],
            inclination_deg=params["inclination_deg"],
            perigee_km=params["perigee_km"],
            apogee_km=params["apogee_km"],
            orbit_zone=get_orbit_zone(sat_alt),
            collision_risk=coll_risk
        )

        ml_danger = danger_engine.estimate_danger(danger_feats)
        final_danger = 0.5 * analytic_danger + 0.5 * ml_danger

        obj["collision_risk"] = coll_risk
        obj["danger_score_analytic"] = analytic_danger
        obj["danger_score_ml"] = ml_danger
        obj["danger_score_final"] = final_danger

        results.append(obj)

    return results
