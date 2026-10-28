import asyncio
import io
import logging
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from backend.model import load_model
from backend.gemini import AssessmentError

load_dotenv(Path(__file__).resolve().parent / '.env')

MAX_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000

@asynccontextmanager
async def lifespan(app):
    app.state.model = load_model()
    app.state.inference_lock = asyncio.Semaphore(1)
    yield

app = FastAPI(title='OXI SIGHT Rust Detection', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in os.getenv('CORS_ORIGINS','http://localhost:5173,http://127.0.0.1:5173').split(',') if x.strip()], allow_methods=['GET','POST'], allow_headers=['Content-Type'])

@app.get('/api/health')
def health():
    return {'mode':app.state.model.mode, 'model_version':app.state.model.version}

def decode(data):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {'JPEG','PNG','WEBP'}:
                    raise HTTPException(415, 'Use a JPEG, PNG, or WebP image.')
                if source.width * source.height > Image.MAX_IMAGE_PIXELS:
                    raise HTTPException(413, 'Image exceeds the 20 megapixel limit.')
                source.load()
                image = ImageOps.exif_transpose(source).convert('RGB')
                image.thumbnail((2048,2048))
                return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(413, 'Image exceeds the 20 megapixel limit.')
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(415, 'Image is corrupt or unsupported. Use JPEG, PNG, or WebP.')

@app.post('/api/detect-rust')
async def detect(file: UploadFile = File(...)):
    try:
        if file.content_type not in {'image/jpeg','image/png','image/webp'}:
            raise HTTPException(415, 'Use a JPEG, PNG, or WebP image.')
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, 'Image must be 10 MB or smaller.')
        if not data:
            raise HTTPException(400, 'Image is empty.')
        async with app.state.inference_lock:
            image = await run_in_threadpool(decode, data)
            try:
                prediction = await run_in_threadpool(app.state.model.predict, image)
            except AssessmentError as exc:
                raise HTTPException(503, str(exc)) from None
            except Exception:
                logging.exception('Model inference failed')
                raise HTTPException(503, 'The model could not process this image. Please try again.')
        return {**prediction,'rust_coverage_percent':None,'processed_image_url':None,'model_version':app.state.model.version,'mode':app.state.model.mode}
    finally:
        await file.close()
