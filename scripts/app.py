import base64
import io
import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from gtts import gTTS
import uvicorn


# Build absolute base directory (parent of scripts folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Correct path for CSV dataset file
DATA_PATH = os.path.join(BASE_DIR, "crop_recommendation", "Fertilizer Prediction.csv")


df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()


DEFAULT_DOSAGE = {
    "Urea": "50 kg per acre",
    "DAP": "40 kg per acre",
    "14-35-14": "30 kg per acre",
    "20-20": "30 kg per acre",
    "Organic compost": "2 tons per acre",
}


crop_translation = {
    'hi': {'मक्का': 'maize', 'गन्ना': 'sugarcane', 'कपास': 'cotton', 'तंबाकू': 'tobacco', 'धान': 'paddy', 'जौ': 'barley',
           'गेहूं': 'wheat', 'मिलेट्स': 'millets', 'तेल बीज': 'oil seeds', 'दालें': 'pulses', 'मूंगफली': 'ground nuts',
           'cotton': 'कपास'},
    'bn': {'ভুট্টা': 'maize', 'চিনি আখ': 'sugarcane', 'তুলা': 'cotton', 'তামাক': 'tobacco', 'धान': 'paddy', 'बार्ली': 'barley',
           'গম': 'wheat', 'মিলেটস': 'millets', 'তেল বীজ': 'oil seeds', 'ডাল': 'pulses', 'মাঠবাদাম': 'ground nuts',
           'cotton': 'তুলা'},
    'pa': {  # Punjabi crop names
        'ਮੱਕਾ': 'maize', 'ਗੰਨਾ': 'sugarcane', 'ਕਪਾਹ': 'cotton', 'ਤਮਾਕੂ': 'tobacco', 'ਚੌਲ': 'paddy', 'ਜੌ': 'barley',
        'ਗਹੂੰ': 'wheat', 'ਮਿਲੇਟਸ': 'millets', 'ਤੈਲ ਬੀਜ': 'oil seeds', 'ਦਾਲਾਂ': 'pulses', 'ਮੂੰਗਫਲੀ': 'ground nuts',
        'cotton': 'ਕਪਾਹ'
    },
    'en': {'maize': 'maize', 'sugarcane': 'sugarcane', 'cotton': 'cotton', 'tobacco': 'tobacco', 'paddy': 'paddy',
           'barley': 'barley', 'wheat': 'wheat', 'millets': 'millets', 'oil seeds': 'oil seeds', 'pulses': 'pulses',
           'ground nuts': 'ground nuts'},
}


soil_type_translations = {
    'en': {'Alluvial Soil': 'Alluvial Soil', 'Arid Soil': 'Arid Soil', 'Black Soil': 'Black Soil',
           'Laterite Soil': 'Laterite Soil', 'Mountain Soil': 'Mountain Soil', 'Red Soil': 'Red Soil',
           'Yellow Soil': 'Yellow Soil'},
    'hi': {'Alluvial Soil': 'अल्लुवियल मिट्टी', 'Arid Soil': 'शुष्क मिट्टी', 'Black Soil': 'काली मिट्टी',
           'Laterite Soil': 'लेटेराइट मिट्टी', 'Mountain Soil': 'पर्वतीय मिट्टी', 'Red Soil': 'लाल मिट्टी',
           'Yellow Soil': 'पीली मिट्टी'},
    'bn': {'Alluvial Soil': 'অলুভিয়াল মাটি', 'Arid Soil': 'শুষ্ক মাটি', 'Black Soil': 'কালো মাটি',
           'Laterite Soil': 'ল্যাটারাইট মাটি', 'Mountain Soil': 'পর্বতীয় মাটি', 'Red Soil': 'লাল মাটি',
           'Yellow Soil': 'হলুদ মাটি'},
    'pa': {  # Punjabi soil types
        'Alluvial Soil': 'ਨਦੀ ਦੀ ਮਿੱਟੀ', 'Arid Soil': 'ਸੁੱਕੀ ਮਿੱਟੀ', 'Black Soil': 'ਕਾਲੀ ਮਿੱਟੀ',
        'Laterite Soil': 'ਲੇਟੇਰਾਈਟ ਮਿੱਟੀ', 'Mountain Soil': 'ਪਹਾੜੀ ਮਿੱਟੀ', 'Red Soil': 'ਲਾਲ ਮਿੱਟੀ',
        'Yellow Soil': 'ਪੀਲੀ ਮਿੱਟੀ'
    }
}


soil_moisture_translations = {
    'en': {'dry': 'dry', 'wet': 'wet'},
    'hi': {'dry': 'सूखा', 'wet': 'गीला'},
    'bn': {'dry': 'শুকনো', 'wet': 'ভেজা'},
    'pa': {'dry': 'ਸੁੱਖਾ', 'wet': 'ਗੀਲਾ'},
}


fertilizer_texts = {
    'en': "For crop '{crop}' on soil '{soil}', apply: {quantity} {fertilizer}. See the manual guide for more instructions.",
    'hi': "फसल '{crop}' के लिए मिट्टी '{soil}' में {quantity} {fertilizer} लगाएं। अधिक जानकारी के लिए मैनुअल देखें।",
    'bn': "ফসল '{crop}' এর জন্য মাটি '{soil}' এ {quantity} {fertilizer} ব্যবহার করুন। বিস্তারিত নির্দেশনার জন্য ম্যানুয়াল দেখুন।",
    'pa': "ਫਸਲ '{crop}' ਲਈ ਮਿੱਟੀ '{soil}' ਵਿੱਚ {quantity} {fertilizer} ਲਗਾਓ। ਹੋਰ ਜਾਣਕਾਰੀ ਲਈ ਮੈਨੁਅਲ ਵੇਖੋ।"
}


default_texts = {
    'en': "Apply 2 tons per acre of Organic compost. See the manual guide for more instructions.",
    'hi': "2 टन प्रति एकड़ जैविक खाद डालें। अधिक जानकारी के लिए मैनुअल देखें।",
    'bn': "প্রতি একর ২ টন জৈব সার ব্যবহার করুন। বিস্তারিত নির্দেশনার জন্য ম্যানুয়াল দেখুন।",
    'pa': "ਹਰ ਏਕੜ ਤੇ 2 ਟਨ ਜੈਵਿਕ ਖਾਦ ਲਗਾਓ। ਹੋਰ ਜਾਣਕਾਰੀ ਲਈ ਮੈਨੁਅਲ ਵੇਖੋ।"
}


moisture_instructions = {
    'en': {
        'dry': "Soil is dry. Consider irrigation before applying fertilizers. Avoid overwatering after application to prevent nutrient leaching.",
        'wet': "Soil is wet. Delay fertilizer application until excess moisture reduces. Ensure good drainage to prevent root damage.",
    },
    'hi': {
        'dry': "मिट्टी सूखी है। उर्वरक लगाने से पहले सिंचाई करें। लागू करने के बाद अधिक पानी न दें ताकि पोषक तत्व न बह जाएं।",
        'wet': "मिट्टी गीली है। अतिरिक्त नमी कम होने तक उर्वरक लगाने में देरी करें। अच्छी जल निकासी सुनिश्चित करें।",
    },
    'bn': {
        'dry': "মাটি শুকনো। সার দেওয়ার আগে সেচ করুন। প্রয়োগের পরে অতিরিক্ত পানি দেবেন না যাতে পুষ্টি তরল না হয়।",
        'wet': "মাটি ভিজে আছে। অতিরিক্ত আর্দ্রতা কমার পর সার দিন। ভাল ড্রেনেজ নিশ্চিত করুন।",
    },
    'pa': {
        'dry': "ਮਿੱਟੀ ਸੁੱਕੀ ਹੈ। ਖਾਦ ਲਗਾਉਣ ਤੋਂ ਪਹਿਲਾਂ ਸਿੰਚਾਈ ਕਰੋ। ਲਾਗੂ ਕਰਨ ਤੋਂ ਬਾਅਦ ਜ਼ਿਆਦਾ ਪਾਣੀ ਨਾ ਦਿਓ ਤਾਂ ਜੋ ਪੋਸ਼ਕ ਤੱਤ ਨਾ ਧੋਏ ਜਾਣ।",
        'wet': "ਮਿੱਟੀ ਗੀਲੀ ਹੈ। ਵਾਧੂ ਨਮੀ ਘਟਣ ਤੱਕ ਖਾਦ ਲਗਾਉਣ ਵਿਚ ਦੇਰ ਕਰੋ। ਚੰਗੀ ਨਿਕਾਸੀ ਯਕੀਨੀ ਬਣਾਓ।"
    }
}


ui_texts = {
    'en': {
        'recognized_soil_type': "Recognized soil type:",
        'recognized_soil_moisture': "Recognized soil moisture:",
    },
    'hi': {
        'recognized_soil_type': "पहचानी गई मिट्टी का प्रकार:",
        'recognized_soil_moisture': "पहचानी गई मिट्टी की नमी:",
    },
    'bn': {
        'recognized_soil_type': "চেনাগোলা মাটির ধরন:",
        'recognized_soil_moisture': "চেনাগোলা মাটির আর্দ্রতা:",
    },
    'pa': {
        'recognized_soil_type': "ਪਛਾਣੀ ਗਈ ਮਿੱਟੀ ਦੀ ਕਿਸਮ:",
        'recognized_soil_moisture': "ਪਛਾਣੀ ਗਈ ਮਿੱਟੀ ਦੀ ਨਮੀ:",
    }
}


fertilizer_name_translation_hi = {
    'Urea': ' यूरिया',
    'DAP': 'डीएपी',
    'Organic compost': 'जैविक खाद',
    '14-35-14': '14-35-14',
    '20-20': '20-20',
}


fertilizer_name_translation_bn = {
    'Urea': 'ইউরিয়া',
    'DAP': 'ডিএপি',
    'Organic compost': 'জৈব সার',
    '14-35-14': '14-35-14',
    '20-20': '20-20',
}


fertilizer_name_translation_pa = {
    'Urea': 'ਯੂਰੀਆ',
    'DAP': 'ਡੀਏਪੀ',
    'Organic compost': 'ਜੈਵਿਕ ਖਾਦ',
    '14-35-14': '14-35-14',
    '20-20': '20-20',
}


app = FastAPI()


origins = ["*"]  # For development, allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    language: str  # 'en', 'hi', 'bn', 'pa'
    crop_name: str
    soil_image_base64: str  # base64-encoded image string


class RecommendResponse(BaseModel):
    text: str
    audio_file: str


def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


model_dir = os.path.join(BASE_DIR, "models")
soil_type_model_path = os.path.join(model_dir, "soil_type_classifier.tflite")
soil_moisture_model_path = os.path.join(model_dir, "soil_moisture_classifier.tflite")


soil_type_interpreter = load_tflite_model(soil_type_model_path)
soil_moisture_interpreter = load_tflite_model(soil_moisture_model_path)


soil_type_labels = ['Alluvial_Soil', 'Arid_Soil', 'Black_Soil', 'Laterite_Soil',
                    'Mountain_Soil', 'Red_Soil', 'Yellow_Soil']
soil_moisture_labels = ['dry', 'wet']


def preprocess_image_from_base64(base64_str):
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.astype('float32')
    img_array = np.expand_dims(img_array, 0)
    return img_array


def run_inference(interpreter, input_data):
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    return output


def translate_crop_name(crop_name, lang_code):
    if not crop_name:
        return crop_name
    crop_name_clean = crop_name.lower().strip()
    # Include Punjabi crop translation
    return crop_translation.get(lang_code, {}).get(crop_name_clean, crop_name_clean)


def get_default_dosage(fertilizer_name, lang):
    return DEFAULT_DOSAGE.get(fertilizer_name, default_texts.get(lang, default_texts['en']))


def get_fertilizer_recommendation(soil_type, crop_name, lang='en'):
    soil_type_clean = soil_type.lower().strip()
    crop_name_clean = crop_name.lower().strip()
    if soil_type_clean.endswith(' soil'):
        soil_type_clean = soil_type_clean[:-5].strip()
    match = df[(df['Soil Type'].str.lower() == soil_type_clean) & (df['Crop Type'].str.lower() == crop_name_clean)]
    if match.empty:
        return default_texts.get(lang, default_texts['en'])
    row = match.iloc[0]
    fertilizer = row.get('Fertilizer Name', 'Unknown Fertilizer').strip()
    quantity = row.get('Fertilizer Quantity')
    if not quantity or pd.isna(quantity) or str(quantity).strip() == '':
        quantity = get_default_dosage(fertilizer, lang)
    if lang == 'hi':
        fertilizer_f = fertilizer_name_translation_hi.get(fertilizer, fertilizer)
        crop_f = translate_crop_name(crop_name, lang)
        soil_f = soil_type_translations[lang].get(soil_type, soil_type)
    elif lang == 'bn':
        fertilizer_f = fertilizer_name_translation_bn.get(fertilizer, fertilizer)
        crop_f = translate_crop_name(crop_name, lang)
        soil_f = soil_type_translations[lang].get(soil_type, soil_type)
    elif lang == 'pa':
        fertilizer_f = fertilizer_name_translation_pa.get(fertilizer, fertilizer)
        crop_f = translate_crop_name(crop_name, lang)
        soil_f = soil_type_translations[lang].get(soil_type, soil_type)
    else:
        fertilizer_f = fertilizer
        crop_f = crop_name.capitalize()
        soil_f = soil_type.capitalize()
    template = fertilizer_texts.get(lang, fertilizer_texts['en'])
    return template.format(crop=crop_f, soil=soil_f, quantity=quantity, fertilizer=fertilizer_f)


def generate_tts_audio(text, lang_code):
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    tts_lang = 'en' if lang_code == 'bn' else lang_code
    tts = gTTS(text=text, lang=tts_lang)
    tts.save(filename)
    return filename


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    lang_code = req.language.lower()
    if lang_code not in ['en', 'hi', 'bn', 'pa']:
        raise HTTPException(status_code=400, detail="Unsupported language code")
    try:
        input_data = preprocess_image_from_base64(req.soil_image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    soil_type_pred = run_inference(soil_type_interpreter, input_data)
    soil_moisture_pred = run_inference(soil_moisture_interpreter, input_data)
    soil_type_raw = soil_type_labels[np.argmax(soil_type_pred)]
    soil_type = soil_type_raw.replace('_', ' ')
    if soil_type.lower().endswith(' soil'):
        soil_type = soil_type[:-5].strip()
    soil_moisture = soil_moisture_labels[np.argmax(soil_moisture_pred)]
    crop_name_translated = translate_crop_name(req.crop_name, lang_code)
    soil_type_print = soil_type_translations.get(lang_code, soil_type_translations['en']).get(soil_type, soil_type)
    soil_moisture_print = soil_moisture_translations.get(lang_code, soil_moisture_translations['en']).get(soil_moisture, soil_moisture)
    ui_text = ui_texts.get(lang_code, ui_texts['en'])
    output_text = (f"{ui_text['recognized_soil_type']} {soil_type_print}.\n"
                   f"{ui_text['recognized_soil_moisture']} {soil_moisture_print}.")
    moist_instr = moisture_instructions.get(lang_code, moisture_instructions['en']).get(soil_moisture, "")
    fertilizer_recommendation = get_fertilizer_recommendation(soil_type, crop_name_translated, lang_code)
    full_text = f"{output_text}\n{moist_instr}\n{fertilizer_recommendation}"
    audio_file = generate_tts_audio(full_text, lang_code)
    return RecommendResponse(text=full_text, audio_file=audio_file)


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    filepath = os.path.join(os.getcwd(), filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="audio/mpeg")
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")


if __name__ == "__main__":
    uvicorn.run("scripts.app:app", host="0.0.0.0", port=8000, reload=True)

