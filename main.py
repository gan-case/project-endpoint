from fastapi import FastAPI, Request, WebSocket
# from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
import uuid
import os
import base64
import json
import httpx
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# from configs.prepare_env import download_all_files
# from similarity_finder.get_similar_images import get_similar_images

FACEMORPH_API_URL = "https://api.facemorph.me/api"
FACEMORPH_ENCODE_IMAGE = "/encodeimage"
FACEMORPH_GENERATE_IMAGE = "/face"

EXPERIMENT_DIR = "experiments/"

app = FastAPI()
app.mount("/assets", StaticFiles(directory="assets", html=True), name="assets")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
app.mount("/experiments", StaticFiles(directory="experiments",
          html=True), name="experiments")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def prepare_env():
    return 0


async def img_format(image, uuid):
    data = {'usrimg': open(image, 'rb')}
    j = {'tryalign': True}

    async with httpx.AsyncClient(follow_redirects=True) as client:
        r = await client.post(FACEMORPH_API_URL +
                              FACEMORPH_ENCODE_IMAGE, files=data, data=j)
        rjson = r.text
        values = {'guid': json.loads(rjson)['guid']}

        r = await client.get(FACEMORPH_API_URL + FACEMORPH_GENERATE_IMAGE, params=values)
        rawimg = r.content
        with open(uuid + "/preprocessed_uploaded_image.jpeg", 'wb') as out_file:
            out_file.write(rawimg)

    return uuid + "/preprocessed_uploaded_image.jpeg"


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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
        data = data.replace('data:image/jpeg;base64,', '')
        uploaded_image.write(base64.b64decode(data))
        uploaded_image.close()
        await websocket.send_json({"status_code": 2, "exp_uuid": exp_uuid})

        await img_format(exp_dir + "/uploaded_image.jpeg", exp_dir)
        await websocket.send_json({"status_code": 3, "exp_uuid": exp_uuid})

        # similar_images = get_similar_images(t, exp_dir + "/uploaded_image.jpeg")
        # await websocket.send_json({"status_code": 4, "exp_uuid": exp_uuid, images: similar_images})

        await websocket.send_json({"status_code": 5, "exp_uuid": exp_uuid, images: []})

        # Completed
        await websocket.send_json({"status_code": 6, "exp_uuid": exp_uuid})


if __name__ == '__main__':
    # download_all_files()

    # global t.load("preprocessed_files/CACD2000_refined_images_embeddings_clusters.ann")

    uvicorn.run("main:app")
