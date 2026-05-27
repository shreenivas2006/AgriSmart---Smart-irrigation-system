import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 62)
print("   AGRISMART - AI CROP YIELD PREDICTOR")
print("   Region: India | Crops: Rice, Wheat, Maize")
print("   Algorithm: Random Forest Regressor")
print("=" * 62)

# ── Load & Filter Dataset ─────────────────────────────────────
df = pd.read_csv("yield_df.csv.xls")
df = df.dropna()

CROPS  = ['Rice, paddy', 'Wheat', 'Maize']
india  = df[(df['Area'] == 'India') & (df['Item'].isin(CROPS))]

print(f"\n  Dataset : {len(india)} records | India only")
print(f"  Crops   : Rice (Paddy), Wheat, Maize")
print(f"  Years   : {india['Year'].min()} – {india['Year'].max()}")

# ── Train One Model Per Crop ──────────────────────────────────
FEATURES = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
models   = {}
accuracy = {}

print("\n  Training AI models...")
for crop in CROPS:
    c = india[india['Item'] == crop]
    X = c[FEATURES]
    y = c['hg/ha_yield']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    m = RandomForestRegressor(n_estimators=100, random_state=42)
    m.fit(X_train, y_train)
    sc = r2_score(y_test, m.predict(X_test))
    models[crop]   = m
    accuracy[crop] = round(sc * 100, 1)
    print(f"    {crop:<15} → {accuracy[crop]}% accuracy")

# ── Pump Decision (from Arduino sensors) ─────────────────────
def pump_decision(moisture, temp):
    if moisture < 30:
        return "PUMP ON  ⚠️  - Critical: soil very dry"
    elif moisture < 40:
        return "PUMP ON  💧 - Soil dry, irrigating"
    elif temp > 35:
        return "PUMP ON  🌡️  - High temp, cooling crop"
    else:
        return "PUMP OFF ✅ - Conditions healthy"

# ── Grade Helper ──────────────────────────────────────────────
def get_grade(val):
    if val >= 35000: return "EXCELLENT 🌾"
    if val >= 25000: return "GOOD ✅"
    if val >= 15000: return "AVERAGE ⚠️"
    return "POOR ❌"

# ── Main Input Loop ───────────────────────────────────────────
print("\n" + "=" * 62)
print("  LIVE SENSOR & FIELD INPUT")
print("  [Read Moisture & Temp from Arduino Serial Monitor]")
print("=" * 62)

moisture = float(input("\n  Soil Moisture % (from Arduino A0)      : "))
temp     = float(input("  Temperature °C  (from Arduino A1)      : "))
rainfall = float(input("  Rainfall mm this week (manual/gauge)   : "))
pest_kg  = float(input("  Pesticides used this season (kg)       : "))

print("\n  Select Crop:")
print("    1. Rice (Paddy)")
print("    2. Wheat")
print("    3. Maize")
choice = input("  Enter 1, 2 or 3 : ").strip()
crop_map  = {'1': 'Rice, paddy', '2': 'Wheat', '3': 'Maize'}
crop_name = crop_map.get(choice, 'Wheat')

# Convert inputs for model
annual_rain = rainfall * 52   # weekly → annual
model_input = [[annual_rain, pest_kg, temp]]

predicted_hg = models[crop_name].predict(model_input)[0]
# Add this "Stress Factor" logic
stress_factor = 1.0
if moisture < 20: stress_factor -= 0.3  # 30% yield loss due to drought
if temp > 40:     stress_factor -= 0.2  # 20% yield loss due to heat stress

predicted_hg = predicted_hg * stress_factor
predicted_kg = round(predicted_hg / 10, 1)   # hg/ha → kg/ha approx
grade        = get_grade(predicted_hg)
pump         = pump_decision(moisture, temp)

# ── Final Output ──────────────────────────────────────────────
print("\n" + "=" * 62)
print("  AGRISMART SYSTEM RESULTS")
print("=" * 62)
print(f"\n  Sensor Readings (from Arduino):")
print(f"    Soil Moisture  : {moisture}%")
print(f"    Temperature    : {temp}°C")
print(f"\n  Field Inputs (manual):")
print(f"    Rainfall       : {rainfall} mm/week  ({int(annual_rain)} mm/year)")
print(f"    Pesticides     : {pest_kg} kg")
print(f"\n  Selected Crop  : {crop_name}")
print(f"\n  PUMP STATUS    : {pump}")
print(f"\n  AI Yield Prediction (Random Forest):")
print(f"    Predicted Yield : {int(predicted_hg):,} hg/ha")
print(f"    In kg/hectare   : {int(predicted_kg):,} kg/ha")
print(f"    Crop Grade      : {grade}")
print(f"    Model Accuracy  : {accuracy[crop_name]}%")
print(f"\n  Model trained on {len(india)} real India crop records")
print("=" * 62)
print("  Refresh (Run again) to enter new sensor values.")
print("=" * 62)