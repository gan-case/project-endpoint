from fastapi import FastAPI, WebSocket
# from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import base64
import json
import httpx
import uvicorn

from configs.prepare_env import download_all_files
from similarity_finder.get_similar_images import get_similar_images

FACEMORPH_API_URL = "https://api.facemorph.me/api"
FACEMORPH_ENCODE_IMAGE = "/encodeimage/"
FACEMORPH_GENERATE_IMAGE = "/face/"

EXPERIMENT_DIR = "experiments/"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory="assets", html=True), name="assets")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.mount("/experiments", StaticFiles(directory="experiments",
          html=True), name="experiments")

test_html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Test</title>
    </head>
    <body>
        <h1>GAN Case Lite</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="file" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };

            async function sendMessage(event) {
                var input = document.getElementById("messageText")
                                  var filereader = new FileReader();
  filereader.readAsDataURL(input.files[0]);
  filereader.onload = function (evt) {
     var base64 = evt.target.result;
    ws.send(base64)
  }
//                 input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""

async def img_format(image, uuid):
    data = {'usrimg': open(image, 'rb')}
    j = {'tryalign': True}

    async with httpx.AsyncClient() as client:
        r = await client.post(FACEMORPH_API_URL +
                              FACEMORPH_ENCODE_IMAGE, files=data, data=j)
        
        rjson = json.loads(r.text)
        values = {'guid': rjson['guid']}
        print(values)
    
        r = await client.get(FACEMORPH_API_URL + FACEMORPH_GENERATE_IMAGE, params=values)
        rawimg = r.content
        with open(uuid + "/preprocessed_uploaded_image.jpeg", 'wb') as out_file:
            out_file.write(rawimg)

    return uuid + "/preprocessed_uploaded_image.jpeg"


@app.get("/")
async def get():
    #return FileResponse("index.html")
    return "ok"


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        # await websocket.send_text(f"Message text was: {data}")

        # Initial Setup
        exp_uuid = str(uuid.uuid4())
        exp_dir = EXPERIMENT_DIR + exp_uuid
        os.makedirs(exp_dir)
        await websocket.send_json({"status_code": 1, "exp_uuid": exp_uuid})

        # Process uploaded image
        uploaded_image = open(exp_dir + "/uploaded_image.jpeg", "wb")
        tmp = data
        data = data.replace('data:image/jpeg;base64,', '')
        uploaded_image.write(base64.b64decode(data))
        uploaded_image.close()
        await websocket.send_json({"status_code": 2, "exp_uuid": exp_uuid, "image": tmp})

        # preprocess: facemorph
        await img_format(exp_dir + "/uploaded_image.jpeg", exp_dir)
        preprocessed_file = open(exp_dir + "/preprocessed_uploaded_image.jpeg", "rb").read()
        base64_utf8_str = base64.b64encode(preprocessed_file).decode('utf-8')
        dataurl = f'data:image/jpeg;base64,{base64_utf8_str}'
        await websocket.send_json({"status_code": 3, "exp_uuid": exp_uuid, "image": dataurl})

        # get similar faces
        similar_images = get_similar_images(exp_dir + "/preprocessed_uploaded_image.jpeg")
        similar_images.pop(0)
        similar_images.pop()
        similar_images_encoded = []
        for image in similar_images:
            similar_image_data = open("dataset/CACD2000/" + image, "rb").read()
            base64_utf8_str = base64.b64encode(similar_image_data).decode('utf-8')
            dataurl = f'data:image/jpeg;base64,{base64_utf8_str}'
            similar_images_encoded.append({"imagename": image, "encodedstring": dataurl})      
        await websocket.send_json({"status_code": 4, "exp_uuid": exp_uuid, "images": similar_images_encoded})

        # run sam
        #run()
        await websocket.send_json({"status_code": 5, "exp_uuid": exp_uuid, "images": []})

        # Completed
        await websocket.send_json({"status_code": 6, "exp_uuid": exp_uuid})
        

if __name__ == '__main__':
    download_all_files()
    uvicorn.run("main:app")
