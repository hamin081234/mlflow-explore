import bentoml
import requests
from pathlib import Path
import io
from PIL import Image

img = Image.open("bRXmz1.jpg").convert("RGB")
print(img)
# print(io.BytesIO(img_data).getvalue())
# print(Image.open(io.BytesIO(img_data)))

# curl -i -X POST "http://localhost:3000/generate_latex" -H "Content-Type: text/plain" -D "bRXmz1.jpg"
answer = requests.post(
    "http://localhost:3000/generate_latex",
    # files = {"upload_file": open('bRXmz1.jpg', 'rb')},
    # files = {"upload_file": open('bRXmz1.jpg', 'rb').read()},
    # files = {"upload_file": io.BytesIO(img_data)},
    # files = {"upload_file": 'bRXmz1.jpg'},
    # headers={"content-type": "image/jpeg"},
    headers={"content-type": "text/plain"},
    data = 'images/스크린샷 2024-03-20 161431.png'
    # data = 'bRXmz1.jpg'
).json()

print("latex: ", answer['latex'])
print("duration: ", answer['duration'])


# with bentoml.SyncHTTPClient("http://localhost:3000") as client:
#     result = client.generate_latex(
#         image=Path("bRXmz1.jpg"),
#         seed=0,
#     )
