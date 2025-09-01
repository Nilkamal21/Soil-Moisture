import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import speech_recognition as sr
from gtts import gTTS
import tempfile
from playsound import playsound
import uuid

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

crop_translation = {
    'hi': {
        'मक्का': 'maize',
        'गन्ना': 'sugarcane',
        'कपास': 'cotton',
        'तंबाकू': 'tobacco',
        'धान': 'paddy',
        'जौ': 'barley',
        'गेहूं': 'wheat',
        'मिलेट्स': 'millets',
        'तेल बीज': 'oil seeds',
        'दालें': 'pulses',
        'मूंगफली': 'ground nuts',
        'cotton': 'कपास'
    },
    'bn': {
        'ভুট্টা': 'maize',
        'চিনি আখ': 'sugarcane',
        'তুলা': 'cotton',
        'তামাক': 'tobacco',
        'ধান': 'paddy',
        'বার্লি': 'barley',
        'গম': 'wheat',
        'মিলেটস': 'millets',
        'তেল বীজ': 'oil seeds',
        'ডাল': 'pulses',
        'মাঠবাদাম': 'ground nuts',
        'cotton': 'তুলা'
    },
    'pa': {  # Punjabi translations
        'ਮੱਕਾ': 'maize',
        'ਗੰਨਾ': 'sugarcane',
        'ਕਪਾਹ': 'cotton',
        'ਤਮਾਕੂ': 'tobacco',
        'ਚაუჳਲ': 'paddy',
        'ਜੌ': 'barley',
        'ਗਹੂੰ': 'wheat',
        'ਮਿਲੇਟਸ': 'millets',
        'ਤੈਲ ਬੀਜ': 'oil seeds',
        'ਦਾਲਾਂ': 'pulses',
        'ਮੂੰਗਫਲੀ': 'ground nuts',
        'cotton': 'ਕਪਾਹ'
    },
    'en': {
        'maize': 'maize',
        'sugarcane': 'sugarcane',
        'cotton': 'cotton',
        'tobacco': 'tobacco',
        'paddy': 'paddy',
        'barley': 'barley',
        'wheat': 'wheat',
        'millets': 'millets',
        'oil seeds': 'oil seeds',
        'pulses': 'pulses',
        'ground nuts': 'ground nuts',
    }
}

soil_type_translations = {
    'en': {
        'Alluvial Soil': 'Alluvial Soil',
        'Arid Soil': 'Arid Soil',
        'Black Soil': 'Black Soil',
        'Laterite Soil': 'Laterite Soil',
        'Mountain Soil': 'Mountain Soil',
        'Red Soil': 'Red Soil',
        'Yellow Soil': 'Yellow Soil'
    },
    'hi': {
        'Alluvial Soil': 'अल्लुवियल मिट्टी',
        'Arid Soil': 'शुष्क मिट्टी',
        'Black Soil': 'काली मिट्टी',
        'Laterite Soil': 'लेटेराइट मिट्टी',
        'Mountain Soil': 'पर्वतीय मिट्टी',
        'Red Soil': 'लाल मिट्टी',
        'Yellow Soil': 'पीली मिट्टी'
    },
    'bn': {
        'Alluvial Soil': 'অলুভিয়াল মাটি',
        'Arid Soil': 'শুষ্ক মাটি',
        'Black Soil': 'কালো মাটি',
        'Laterite Soil': 'ল্যাটারাইট মাটি',
        'Mountain Soil': 'পর্বতীয় মাটি',
        'Red Soil': 'লাল মাটি',
        'Yellow Soil': 'হলুদ মাটি'
    },
    'pa': {  # Punjabi soil types
        'Alluvial Soil': 'ਨਦੀ ਦੀ ਮਿੱਟੀ',
        'Arid Soil': 'ਸੁੱਕੀ ਮਿੱਟੀ',
        'Black Soil': 'ਕਾਲੀ ਮਿੱਟੀ',
        'Laterite Soil': 'ਲੇਟੇਰਾਈਟ ਮਿੱਟੀ',
        'Mountain Soil': 'ਪਹਾੜੀ ਮਿੱਟੀ',
        'Red Soil': 'ਲਾਲ ਮਿੱਟੀ',
        'Yellow Soil': 'ਪੀਲੀ ਮਿੱਟੀ'
    }
}

soil_moisture_translations = {
    'en': {
        'dry': 'dry',
        'wet': 'wet'
    },
    'hi': {
        'dry': 'सूखा',
        'wet': 'गीला'
    },
    'bn': {
        'dry': 'শুকনো',
        'wet': 'ভেজা'
    },
    'pa': {  # Punjabi moisture
        'dry': 'ਸੁੱਖਾ',
        'wet': 'ਗੀਲਾ'
    }
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
        'wet': "Soil is wet. Delay fertilizer application until excess moisture reduces. Ensure good drainage to prevent root damage."
    },
    'hi': {
        'dry': "मिट्टी सूखी है। उर्वरक लगाने से पहले सिंचाई करें। लागू करने के बाद अधिक पानी न दें ताकि पोषक तत्व न बह जाएं।",
        'wet': "मिट्टी गीली है। अतिरिक्त नमी कम होने तक उर्वरक लगाने में देरी करें। अच्छी जल निकासी सुनिश्चित करें।"
    },
    'bn': {
        'dry': "মাটি শুকনো। সার দেওয়ার আগে সেচ করুন। প্রয়োগের পরে অতিরিক্ত পানি দেবেন না যাতে পুষ্টি তরল না হয়।",
        'wet': "মাটি ভিজে আছে। অতিরিক্ত আর্দ্রতা কমার পর সার দিন। ভাল ড্রেনেজ নিশ্চিত করুন।"
    },
    'pa': {
        'dry': "ਮਿੱਟੀ ਸੁੱਕੀ ਹੈ। ਖਾਦ ਲਗਾਉਣ ਤੋਂ ਪਹਿਲਾਂ ਸਿੰਚਾਈ ਕਰੋ। ਲਾਗੂ ਕਰਨ ਤੋਂ ਬਾਅਦ ਜ਼ਿਆਦਾ ਪਾਣੀ ਨਾ ਦਿਓ ਤਾਂ ਜੋ ਪੋਸ਼ਕ ਤੱਤ ਨਾ ਧੋਏ ਜਾਣ।",
        'wet': "ਮਿੱਟੀ ਗੀਲੀ ਹੈ। ਵਾਧੂ ਨਮੀ ਘਟਣ ਤੱਕ ਖਾਦ ਲਗਾਉਣ ਵਿਚ ਦੇਰ ਕਰੋ। ਚੰਗੀ ਨਿਕਾਸੀ ਯਕੀਨੀ ਬਣਾਓ।"
    }
}

ui_texts = {
    'en': {
        'select_language': "Select language:",
        'lang_options': "1. English\n2. Hindi / हिंदी\n3. Bangla / বাংলা\n4. Punjabi / ਪੰਜਾਬੀ",
        'enter_choice': "Enter choice (1/2/3/4): ",
        'upload_capture': "Enter '1' to upload image, '2' to capture image via camera: ",
        'speak_type': "Enter '1' to speak crop name, '2' to type crop name: ",
        'say_crop': "Please say the crop name:",
        'recognized_crop': "Recognized crop name:",
        'recognized_soil_type': "Recognized soil type:",
        'recognized_soil_moisture': "Recognized soil moisture:",
        'typed': "Entered crop name:",
        'confirm_continue': "Press Enter to confirm and continue, or Ctrl+C to retry...",
        'cannot_understand': "Sorry, could not understand. Please try again.",
        'try_again': "Please try speaking again.",
        'crop_empty': "Crop name cannot be empty.",
        'invalid_choice': "Invalid choice, please try again.",
        'no_file_selected': "No file selected. Exiting.",
        'no_image_captured': "No image captured. Exiting.",
        'invalid_option': "Invalid option. Exiting.",
    },
    'hi': {
        'select_language': "भाषा चुनें:",
        'lang_options': "1. अंग्रेज़ी\n2. हिंदी\n3. बंगला\n4. पंजाबी",
        'enter_choice': "चुनाव दर्ज करें (1/2/3/4): ",
        'upload_capture': "अपलोड के लिए '1' दबाएं, कैमरा से फोटो लेने के लिए '2' दबाएं: ",
        'speak_type': "फसल का नाम बोलने के लिए '1' दबाएं, टाइप करने के लिए '2' दबाएं: ",
        'say_crop': "कृपया फसल का नाम बोलें:",
        'recognized_crop': "पहचाना गया फसल नाम:",
        'recognized_soil_type': "पहचानी गई मिट्टी का प्रकार:",
        'recognized_soil_moisture': "पहचानी गई मिट्टी की नमी:",
        'typed': "दर्ज किया गया फसल नाम:",
        'confirm_continue': "जारी रखने के लिए Enter दबाएं, या दोबारा प्रयास के लिए Ctrl+C दबाएं...",
        'cannot_understand': "क्षमा करें, समझ नहीं पाया। कृपया पुनः प्रयास करें।",
        'try_again': "कृपया पुनः बोलकर प्रयास करें।",
        'crop_empty': "फसल का नाम खाली नहीं हो सकता।",
        'invalid_choice': "अमान्य विकल्प, कृपया पुनः प्रयास करें।",
        'no_file_selected': "कोई फ़ाइल चयनित नहीं है। कार्यक्रम बंद किया जा रहा है।",
        'no_image_captured': "कोई छवि कैप्चर नहीं हुई। कार्यक्रम बंद किया जा रहा है।",
        'invalid_option': "अमान्य विकल्प। कार्यक्रम बंद किया जा रहा है।",
    },
    'bn': {
        'select_language': "ভাষা নির্বাচন করুন:",
        'lang_options': "1. ইংরেজি\n2. হিন্দি\n3. বাংলা\n4. পাঞ্জাবি",
        'enter_choice': "পছন্দ লিখুন (1/2/3/4): ",
        'upload_capture': "আপলোড করতে '1' চাপুন, ক্যামেরা থেকে ছবি নিতে '2' চাপুন: ",
        'speak_type': "ফসলের নাম বলার জন্য '1' চাপুন, টাইপ করার জন্য '2' চাপুন: ",
        'say_crop': "ফসলের নাম বলুন:",
        'recognized_crop': "চেনাগোলা ফসলের নাম:",
        'recognized_soil_type': "চেনাগোলা মাটির ধরন:",
        'recognized_soil_moisture': "চেনাগোলা মাটির আর্দ্রতা:",
        'typed': "টাইপ করা ফসলের নাম:",
        'confirm_continue': "অগ্রসর হতে Enter চাপুন, পুনরায় চেষ্টা করতে Ctrl+C চাপুন...",
        'cannot_understand': "দুঃখিত, বুঝতে পারিনি। আবার চেষ্টা করুন।",
        'try_again': "আবার বলার চেষ্টা করুন।",
        'crop_empty': "ফসলের নাম খালি হতে পারে না।",
        'invalid_choice': "অবৈধ নির্বাচন, আবার চেষ্টা করুন।",
        'no_file_selected': "কোনো ফাইল নির্বাচন করা হয়নি। প্রস্থান করা হচ্ছে।",
        'no_image_captured': "কোনো ছবি ধারণ করা হয়নি। প্রস্থান করা হচ্ছে।",
        'invalid_option': "অবৈধ অপশন। প্রস্থান করা হচ্ছে।",
    },
    'pa': {
        'select_language': "ਭਾਸ਼ਾ ਚੁਣੋ:",
        'lang_options': "1. ਅੰਗਰੇਜ਼ੀ\n2. ਹਿੰਦੀ\n3. ਬੰਗਲਾ\n4. ਪੰਜਾਬੀ",
        'enter_choice': "ਚੋਣ ਦਰਜ ਕਰੋ (1/2/3/4): ",
        'upload_capture': "ਤਸਵੀਰ ਅਪਲੋਡ ਕਰਨ ਲਈ '1' ਦਬਾਓ, ਕੈਮਰਾ ਰਾਹੀਂ ਤਸਵੀਰ ਖਿੱਚਣ ਲਈ '2' ਦਬਾਓ: ",
        'speak_type': "ਫਸਲ ਦਾ ਨਾਮ ਬੋਲਣ ਲਈ '1' ਦਬਾਓ, ਟਾਈਪ ਕਰਨ ਲਈ '2' ਦਬਾਓ: ",
        'say_crop': "ਕਿਰਪਾ ਕਰਕੇ ਫਸਲ ਦਾ ਨਾਮ ਬੋਲੋ:",
        'recognized_crop': "ਪਛਾਨੀ ਗਈ ਫਸਲ ਦਾ ਨਾਮ:",
        'recognized_soil_type': "ਪਛਾਨੀ ਗਈ ਮਿੱਟੀ ਦੀ ਕਿਸਮ:",
        'recognized_soil_moisture': "ਪਛਾਨੀ ਗਈ ਮਿੱਟੀ ਦੀ ਨਮੀ:",
        'typed': "ਦਰਜ ਕੀਤਾ ਫਸਲ ਦਾ ਨਾਮ:",
        'confirm_continue': "ਜਾਰੀ ਰੱਖਣ ਲਈ Enter ਦਬਾਓ, ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਲਈ Ctrl+C ਦਬਾਓ...",
        'cannot_understand': "ਮਾਫ ਕਰਨਾ, ਸਮਝ ਨਹੀਂ ਆਇਆ। ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
        'try_again': "ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਬੋਲ ਕੇ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
        'crop_empty': "ਫਸਲ ਦਾ ਨਾਮ ਖਾਲੀ ਨਹੀਂ ਹੋ ਸਕਦਾ।",
        'invalid_choice': "ਗਲਤ ਚੋਣ, ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
        'no_file_selected': "ਕੋਈ ਫਾਇਲ ਚੁਣੀ ਨਹੀਂ ਗਈ। ਬਾਹਰ ਨਿਕਲਿਆ ਜਾ ਰਿਹਾ ਹੈ।",
        'no_image_captured': "ਕੋਈ ਤਸਵੀਰ ਨਹੀਂ ਖਿੱਚੀ ਗਈ। ਬਾਹਰ ਨਿਕਲਿਆ ਜਾ ਰਿਹਾ ਹੈ।",
        'invalid_option': "ਗਲਤ ਵਿਕਲਪ। ਬਾਹਰ ਨਿਕਲਿਆ ਜਾ ਰਿਹਾ ਹੈ।",
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

def get_default_dosage(fertilizer_name, lang):
    return DEFAULT_DOSAGE.get(fertilizer_name, default_texts.get(lang, default_texts['en']))

def translate_crop_name(crop_name, lang_code):
    if not crop_name:
        return crop_name
    crop_name_clean = crop_name.lower().strip()
    translations = crop_translation.get(lang_code, {})
    return translations.get(crop_name_clean, crop_name_clean)

def get_fertilizer_recommendation(soil_type, crop_name, lang='en'):
    soil_type_clean = soil_type.strip().lower()
    crop_name_clean = crop_name.strip().lower()

    if soil_type_clean.endswith(' soil'):
        soil_type_clean = soil_type_clean[:-5].strip()

    match = df[
        (df['Soil Type'].str.lower() == soil_type_clean) & 
        (df['Crop Type'].str.lower() == crop_name_clean)
    ]

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

def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_path, target_size=(224,224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)/255.0
    img_array = img_array.astype('float32')
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def run_inference(interpreter, input_data):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    return output

soil_type_labels = ['Alluvial_Soil', 'Arid_Soil', 'Black_Soil', 'Laterite_Soil',
                    'Mountain_Soil', 'Red_Soil', 'Yellow_Soil']
soil_moisture_labels = ['dry', 'wet']

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Soil Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
    )
    return file_path

def capture_image_from_camera(save_path='captured_soil.jpg'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(ui_texts[lang_code]['no_image_captured'])
        return None

    print("Press 'Space' to capture the image, 'Esc' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Capture Soil Image", frame)
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape hit, closing...")
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif k%256 == 32:
            cv2.imwrite(save_path, frame)
            print(f"Image saved to {save_path}")
            cap.release()
            cv2.destroyAllWindows()
            return save_path

def listen_crop_name(stt_lang):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(ui_texts[lang_code]['say_crop'])
        audio = recognizer.listen(source)
    try:
        crop_name = recognizer.recognize_google(audio, language=stt_lang)
        print(f"{ui_texts[lang_code]['recognized_crop']} {crop_name}")
        return crop_name
    except sr.UnknownValueError:
        print(ui_texts[lang_code]['cannot_understand'])
        return None
    except sr.RequestError:
        print("Speech recognition service error.")
        return None

def speak_text(text, lang_code_speech):
    if lang_code_speech.startswith('bn'):
        lang_code_speech = 'en'
    temp_filename = f"tts_{uuid.uuid4().hex}.mp3"
    try:
        tts = gTTS(text=text, lang=lang_code_speech)
        tts.save(temp_filename)
        playsound(temp_filename)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def get_crop_name(stt_lang):
    texts = ui_texts[lang_code]
    while True:
        choice = input(texts['speak_type']).strip()
        if choice == '1':
            crop_name = listen_crop_name(stt_lang)
            if crop_name:
                input(texts['confirm_continue'])
                return crop_name
            else:
                print(texts['try_again'])
        elif choice == '2':
            crop_name = input("Type crop name: ").strip()
            if crop_name:
                print(f"{texts['typed']} {crop_name}")
                input(texts['confirm_continue'])
                return crop_name
            else:
                print(texts['crop_empty'])
        else:
            print(texts['invalid_choice'])

def main():
    global lang_choice, lang_code

    print("Select language / भाषा / ভাষা / ਭਾਸ਼ਾ:")
    print("1. English")
    print("2. Hindi / हिंदी")
    print("3. Bangla / বাংলা")
    print("4. Punjabi / ਪੰਜਾਬੀ")
    lang_choice = input("Enter choice (1/2/3/4): ").strip()

    lang_map = {
        '1': ('en', 'en-US'),
        '2': ('hi', 'hi-IN'),
        '3': ('bn', 'bn-BD'),
        '4': ('pa', 'pa-IN')
    }

    if lang_choice not in lang_map:
        print("Invalid selection, defaulting to English.")
        lang_choice = '1'

    lang_code, stt_lang = lang_map[lang_choice]

    soil_type_interpreter = load_tflite_model('models/soil_type_classifier.tflite')
    soil_moisture_interpreter = load_tflite_model('models/soil_moisture_classifier.tflite')

    texts = ui_texts[lang_code]

    choice = input(texts['upload_capture']).strip()
    if choice == '1':
        image_path = open_file_dialog()
        if not image_path:
            print(texts['no_file_selected'])
            return
    elif choice == '2':
        image_path = capture_image_from_camera()
        if image_path is None:
            print(texts['no_image_captured'])
            return
    else:
        print(texts['invalid_option'])
        return

    crop_name = get_crop_name(stt_lang)
    crop_name_translated = translate_crop_name(crop_name, lang_code)

    input_data = preprocess_image(image_path)
    soil_type_pred = run_inference(soil_type_interpreter, input_data)
    soil_moisture_pred = run_inference(soil_moisture_interpreter, input_data)

    soil_type_raw = soil_type_labels[np.argmax(soil_type_pred)]
    soil_type = soil_type_raw.replace('_', ' ')
    if soil_type.lower().endswith(' soil'):
        soil_type = soil_type[:-5].strip()

    soil_moisture = soil_moisture_labels[np.argmax(soil_moisture_pred)]

    soil_type_print = soil_type_translations.get(lang_code, soil_type_translations['en']).get(soil_type, soil_type)
    soil_moisture_print = soil_moisture_translations.get(lang_code, soil_moisture_translations['en']).get(soil_moisture, soil_moisture)

    output_text = f"{texts['recognized_soil_type']} {soil_type_print}.\n{texts['recognized_soil_moisture']} {soil_moisture_print}."

    moist_instr = moisture_instructions.get(lang_code, moisture_instructions['en']).get(soil_moisture, "")

    fertilizer_recommendation = get_fertilizer_recommendation(soil_type, crop_name_translated, lang_code)

    full_text = f"{output_text}\n{moist_instr}\n{fertilizer_recommendation}"
    print("\n" + full_text)

    speak_text(full_text, lang_code)

if __name__ == "__main__":
    main()
