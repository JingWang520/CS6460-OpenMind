
import requests

def test_generate_text():
    url = "http://10.25.10.144:8008/generate-text/"
    payload = {
        "messages": [
            {"role": "user", "content": "Hello, who are you?"}
        ]
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        print("Response from /generate-text/:")
        print(data["response"])
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_generate_text()
