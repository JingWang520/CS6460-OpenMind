# test_tts.py
import requests
import base64

url = "http://10.25.10.144:8008/tts/"

payload = {
    "text": "Hello, this is a test of the text-to-speech system."
}

response = requests.post(url, json=payload)
if response.status_code == 200:
    data = response.json()
    audio_base64 = data['audio_base64']
    audio_data = base64.b64decode(audio_base64)
    with open("test_output_tts.wav", "wb") as f:
        f.write(audio_data)
    print("Audio saved to test_output_tts.wav")
else:
    print("Failed to generate speech:", response.text)
