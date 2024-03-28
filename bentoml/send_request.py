import requests
import json

# curl -i -X POST "http://localhost:3000/generate_latex" -f "image=@./images/a.jpg" (X)
# curl -i -X POST "http://localhost:3000/generate_latex" -F "image=@/mnt/c/Users/hamin/Documents/pythonProject/mlflow-explore/bentoml/images/a.jpg" (O)
file_path = 'images'
file_name = 'a.jpg'
answer = requests.post(
    "http://localhost:3000/generate_latex",
    files = {"upload_file": open('/'.join([file_path, file_name]), 'rb')},
    # headers = {'Content-Type':"application/json"},
).text

print(json.loads(answer))
