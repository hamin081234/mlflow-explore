import requests
import json
import torch

# curl -i -X POST "http://localhost:3000/generate_latex" -f "image=@./images/a.jpg" (X)
# curl -i -X POST "http://localhost:3000/generate_latex" -F "image=@/mnt/c/Users/hamin/Documents/pythonProject/mlflow-explore/bentoml/images/a.jpg" (O)
# curl -i -X POST "http://localhost:3000/generate_latex" -F "image=@"C:\Users\user\Pictures\Screenshots\a.png" (O)
# curl -i -X POST "http://localhost:3000/generate_latex" -H 'accept: application/json' -H 'Content-Type: image/png' --data-binary "@/mnt/c/Users/user/Desktop/hama/mlflow-explore/bentoml_donut/images/a.jpg" (O)

file_path = 'images'
file_name = 'b.png'
with open('/'.join([file_path, file_name]), 'rb') as f:
    data = f.read()

answer = requests.post(
    "http://localhost:3000/generate_latex",
    data = data,
    headers = {"Content-Type": "image/png", "accept": "application/json"},
).text

print(json.loads(answer))

"""
docker run --gpus all --rm -p 3000:3000 image2latex:l7mdj7xmwkxquaav
"""
