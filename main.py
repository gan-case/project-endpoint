from fastapi import FastAPI, WebSocket
# from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import base64
from ast import literal_eval
import json
import logging
import httpx
import gc
from subprocess import Popen
import uvicorn
import pandas as pd
import shutil
from annoy import AnnoyIndex
from inference_script import run
from gradio_client import Client
from morph import morph_images
from similarity_finder.morph_similar_images import get_morphed_images

from configs.prepare_env import download_all_files
#from similarity_finder.get_similar_images import get_similar_images

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

client = Client("https://99ashutosh-find-similar-image.hf.space/--replicas/p86sv/")

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

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(FACEMORPH_API_URL +
                              FACEMORPH_ENCODE_IMAGE, files=data, data=j)
        
        rjson = json.loads(r.text)
        values = {'guid': rjson['guid']}
    
        r = await client.get(FACEMORPH_API_URL + FACEMORPH_GENERATE_IMAGE, params=values)
        rawimg = r.content
        with open(uuid + "/preprocessed_uploaded_image.jpeg", 'wb') as out_file:
            out_file.write(rawimg)


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
        exp_dir = EXPERIMENT_DIR + exp_uuid + "/raw_images"
        os.makedirs(exp_dir)
        await websocket.send_json({"status_code": 1, "exp_uuid": exp_uuid})

        # Process uploaded image
        uploaded_image = open(exp_dir + "/../uploaded_image.jpeg", "wb")
        tmp = data
        data = data.replace('data:image/jpeg;base64,', '')
        uploaded_image.write(base64.b64decode(data))
        uploaded_image.close()
        await websocket.send_json({"status_code": 2, "exp_uuid": exp_uuid, "image": tmp})

        # preprocess: facemorph
        await img_format(exp_dir + "/../uploaded_image.jpeg", exp_dir)
        preprocessed_file = open(exp_dir + "/preprocessed_uploaded_image.jpeg", "rb").read()
        base64_utf8_str = base64.b64encode(preprocessed_file).decode('utf-8')
        dataurl = f'data:image/jpeg;base64,{base64_utf8_str}'
        await websocket.send_json({"status_code": 3, "exp_uuid": exp_uuid, "image": dataurl})

        # get similar faces
        #global df
        #t, similar_images = get_similar_images(df, exp_dir + "/preprocessed_uploaded_image.jpeg", 23, "M", "White")
        similar_images = client.predict(
            f'{exp_dir}/preprocessed_uploaded_image.jpeg',
            "23",
            "M",    # str  in 'Gender' Textbox component
            "White",        # str  in 'Race' Textbox component
            api_name="/predict"
        )
        #similar_images.pop(0)
        #similar_images.pop()
        similar_images = literal_eval(similar_images)
        similar_images = similar_images[1:6]
        similar_images_encoded = []
        for image in similar_images:
            similar_image_data = open("dataset/CACD2000/" + image, "rb").read()
            shutil.copyfile("dataset/CACD2000/" + image, exp_dir +"/"+ image)
            base64_utf8_str = base64.b64encode(similar_image_data).decode('utf-8')
            dataurl = f'data:image/jpeg;base64,{base64_utf8_str}'
            similar_images_encoded.append({"imagename": image, "encodedstring": dataurl})      
        await websocket.send_json({"status_code": 4, "exp_uuid": exp_uuid, "images": similar_images_encoded})

        # run morph on top 5
        os.makedirs(exp_dir + "/../morphed_images/")
        morphed_images = await get_morphed_images(exp_dir + "/preprocessed_uploaded_image.jpeg", similar_images, exp_dir + "/../morphed_images/")

        # run sam
        path1 = exp_dir + "/../SAM_outputs"
        path2 = exp_dir + "/../morphed_images/"
        age = ["30", "40", "50", "60", "70"]
        os.system(f"python inference_script.py {path1} {path2}")
        #sam_processed_images = os.listdir(exp_dir + "/../F0")
        #sam_processed_encoded = []
        #for image in sam_processed_images:
        #    sam_processed_data = open(exp_dir + "/../F0/" + image, "rb").read()
        #    base64_utf8_str = base64.b64encode(sam_processed_data).decode('utf-8')
        #    dataurl = f'data:image/jpeg;base64,{base64_utf8_str}'
        #    sam_processed_encoded.append({"imagename": image, "encodedstring": dataurl})      
        #await websocket.send_json({"status_code": 6, "exp_uuid": exp_uuid, "images": sam_processed_encoded})

        # run GA
        #for i in ["30", "40", "50", "60", "70"]:
        await morph_images(exp_dir + "/preprocessed_uploaded_image.jpeg", exp_dir + f'/../SAM_outputs/30/F0/')
        await websocket.send_json({"status_code": 7, "exp_uuid": exp_uuid})

        # Completed
        await websocket.send_json({"status_code": 8, "exp_uuid": exp_uuid})
        

if __name__ == '__main__':
    download_all_files()
    uvicorn.run("main:app")