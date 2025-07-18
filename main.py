from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import torch
import numpy as np
import uuid
import os
import cv2

app = FastAPI()

# Load model once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)

@app.post("/generate/")
async def generate(file: UploadFile = File(...)):
    filename = f"/tmp/{uuid.uuid4()}.mp4"
    img_path = "/tmp/input.png"

    # Save the uploaded image
    with open(img_path, "wb") as f:
        f.write(await file.read())

    image = Image.open(img_path).convert("RGB").resize((1024, 576))
    output = pipe(image, decode_chunk_size=8, num_frames=14)
    frames = output.frames[0]

    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 8, (1024, 576))
    for frame in frames:
        out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    out.release()

    return FileResponse(filename, media_type="video/mp4", filename="output.mp4")
