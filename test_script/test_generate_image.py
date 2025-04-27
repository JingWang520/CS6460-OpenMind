# test_generate_image.py
import requests
import base64

url = "http://10.25.10.144:8008/generate-image/"

payload = {
    "prompt": "a futuristic cityscape with flying cars"
}

response = requests.post(url, json=payload)
if response.status_code == 200:
    data = response.json()
    print("Image generated successfully. Saving image...")
    img_base64 = data['image']
    img_data = base64.b64decode(img_base64)
    with open("test_output_image.png", "wb") as f:
        f.write(img_data)
    print("Image saved to test_output_image.png")
else:
    print("Failed to generate image:", response.text)
