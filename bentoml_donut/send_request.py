import requests
import json
import torch

# curl -i -X POST "http://localhost:3000/generate_latex" -f "image=@./images/a.jpg" (X)
# curl -i -X POST "http://localhost:3000/generate_latex" -F "image=@/mnt/c/Users/hamin/Documents/pythonProject/mlflow-explore/bentoml/images/a.jpg" (O)
# curl -i -X POST "http://localhost:3000/generate_latex" -F "image=@"C:\Users\user\Pictures\Screenshots\a.png" (O)
# file_path = 'images'
# file_name = 'b.png'
# answer = requests.post(
#     "http://localhost:3000/generate_latex",
#     files = {"upload_file": open('/'.join([file_path, file_name]), 'rb')},
#     # headers = {'Content-Type':""},
# ).text

# print(json.loads(answer))


# file_path = 'images'
# file_name = 'b.png'
# with open('/'.join([file_path, file_name]), 'rb') as f:
#     data = f.read()

# answer = requests.post(
#     "http://localhost:3000/generate_latex",
#     data = data,
#     headers = {"Content-Type": "image/png", "accept": "application/json"},
# ).text

# print(json.loads(answer))


# answer = requests.post(
#     "http://localhost:3000/return_text",
#     data = "texttexttexttesttest",
#     headers = {'Content-Type':"text/plain"},
# ).text

# print(answer)

"""
docker run --gpus all --rm -p 3000:3000 image2latex:l7mdj7xmwkxquaav
"""
