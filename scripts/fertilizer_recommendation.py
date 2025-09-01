import pandas as pd

DATA_PATH = "crop_recommendation/Fertilizer Prediction.csv"
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

DEFAULT_DOSAGE = {
    "Urea": "50 kg per acre",
    "DAP": "40 kg per acre",
    "14-35-14": "30 kg per acre",
    "20-20": "30 kg per acre",
    "Organic compost": "2 tons per acre",
}

def get_default_dosage(fertilizer_name):
    return DEFAULT_DOSAGE.get(fertilizer_name, "Use as per packaging instructions")

def get_fertilizer_recommendation(soil_type, crop_name):
    if not soil_type or not crop_name:
        return None

    soil_type = soil_type.strip().lower()
    crop_name = crop_name.strip().lower()

    match = df[
        (df['Soil Type'].str.lower() == soil_type) & 
        (df['Crop Type'].str.lower() == crop_name)
    ]

    if match.empty:
        return ("Apply 2 tons per acre of Organic compost. "
                "See the manual guide for more instructions.")

    row = match.iloc[0]
    fertilizer = row.get('Fertilizer Name', 'Unknown Fertilizer').strip().capitalize()
    quantity = row.get('Fertilizer Quantity')

    if not quantity or pd.isna(quantity) or str(quantity).strip() == '':
        quantity = get_default_dosage(fertilizer)

    return (f"For crop '{crop_name.capitalize()}' on soil '{soil_type.capitalize()}', apply: "
            f"{quantity} {fertilizer}. See the manual guide for more instructions.")

if __name__ == "__main__":
    soil = input("Enter soil type: ").strip()
    crop = input("Enter crop name: ").strip()

    recommendation = get_fertilizer_recommendation(soil, crop)
    if recommendation:
        print(recommendation)
    else:
        print("Please enter both soil type and crop name to get recommendations.")
