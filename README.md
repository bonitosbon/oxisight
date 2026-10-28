# OXI SIGHT / RustScan AI

Mobile-first React + Vite + Tailwind application with a Python FastAPI/OpenCV backend. The supplied dark amber reference screens guide the visual design. Camera capture, confirmed image upload, results, JSON report export, IndexedDB history, and an installable offline application shell are implemented.

**No trained rust-specific model is included.** Without provider credentials or weights, development mode validates images and returns `unable_to_determine`. With Gemini configured, the app performs real, experimental visual assessments for presentations. These are not validated rust detections. Confidence, coverage, and overlay remain null. Depth, structural safety, and severity are never measured.

## Gemini presentation demo

Copy `backend/.env.example` to `backend/.env`, set `GEMINI_API_KEY`, and restart the backend. This file is Git-ignored and is loaded only by Python. Never use a `VITE_` variable for the key. `GEMINI_MODEL` defaults to `gemini-2.5-flash`; choose an image-capable model available to your Google AI project if needed. A configured ONNX path takes precedence over Gemini.

The review screen discloses Google processing before Analyze Image. The backend sends a resized JPEG over HTTPS using Gemini's image-understanding and structured-output API. API quota, permission, timeout, and malformed-response failures appear as errors; they never produce substitute detections. Model responses are labeled experimental, and no numerical confidence is fabricated. Google project quota/billing must permit requests for the demo to work.

API references: https://ai.google.dev/gemini-api/docs/image-understanding and https://ai.google.dev/gemini-api/docs/structured-output.

## Run locally

Requires Node 20.19+ and Python 3.11–3.12.

```sh
npm install
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` on macOS/Linux), then:

```sh
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```sh
npm run dev
```

Open http://localhost:5173. Vite proxies `/api` to port 8000. API docs: http://localhost:8000/docs. Use HTTPS when accessing from a phone: mobile browsers do not allow live camera access on plain HTTP LAN addresses. The take-photo/file chooser fallback is available when live camera access fails. Image upload requires tapping **Analyze Image**. Only explicitly saved results are retained locally; server images are processed in memory, with no persistent image files.

## Integrate a real trained model

1. Train and validate a rust classifier on representative labeled metal images. Keep test data independent of training data and evaluate nonmetal/poor-quality inputs as well as rust and clean metal.
2. Export an ONNX model with one float32 input shaped `[1,3,224,224]` and one output containing **two logits**, ordered `[no_rust, rust]`. The included adapter uses RGB, resize, and ImageNet mean/std normalization. Change `backend/model.py` to match the exact training preprocessing, input size, output semantics, and class order. Do not plug arbitrary weights into this contract.
3. Set `RUST_MODEL_PATH` to the model file path and `MODEL_VERSION` to your validated version. Optional `MODEL_INPUT_SIZE` defaults to 224; `MODEL_THRESHOLD` defaults to 0.8. Tune the abstention threshold using held-out validation data.
4. Leave `MODEL_CONFIDENCE_CALIBRATED=false` until calibration has been measured on held-out data. Softmax values are not automatically calibrated confidence. Set it to `true` only after calibration is validated.
5. Restart the backend and test known positive, negative, uncertain, and invalid inputs. `/api/health` reports `mode: model` when weights load; this indicates connection, **not scientific validation**. Incorrect weights fail startup rather than falling back silently.

The classifier provides no segmentation, coverage, or overlay. Those fields remain null. For highlighting, replace the adapter with a validated segmentation model and extend the frontend to render its mask. A single image cannot establish structural safety or corrosion depth.

## Tests and build

```sh
python -m pytest backend -q
npm run build
npm run preview
```

API tests exercise a real encoded image through preprocessing and check corrupt, unsupported, empty, and oversized input rejection. The production build generates a service worker and manifest, with 192/512px icons. The offline shell and saved history work after the first successful production visit; inference requires an online backend. Test installation using the production build, not the Vite dev server.

Manual phone acceptance: Safari and Chrome over HTTPS; permission accepted/denied; rear camera; capture/retake; JPEG/PNG/WebP upload; analysis cancel/error; save/reload/delete history; offline relaunch; home-screen installation. Browser storage can be evicted, so export reports when needed. JSON exports contain metadata, not the original photo. Real-device camera and installation behavior must be verified on target phones.

## Deploy frontend to Vercel

Import this repository, select Vite, use `npm run build` and output directory `dist`. Set `VITE_API_URL=https://YOUR-PYTHON-BACKEND` before building. This is a public origin setting, not a secret. The client uses state navigation and needs no route rewrites. HTTPS is required for camera and PWA functionality.

## Deploy Python backend

Use a Python-compatible host such as Render or Railway, from the repository root. Build: `pip install -r backend/requirements.txt`. Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`. Set `CORS_ORIGINS` to the exact Vercel origin (comma-separated for multiple allowed frontends). Configure model environment variables only when validated weights are available. Mount model weights through the hosting provider; they are excluded from Git. Start in development mode to demonstrate the workflow honestly.

Configure host-level request-body limits and rate limiting before exposing the research endpoint publicly. Application validation accepts only actual JPEG/PNG/WebP images, limits decoded dimensions to 20 megapixels, reads at most 10 MB + 1 byte, normalizes EXIF orientation, bounds preprocessing dimensions, and serializes inference per worker. No user filename is used as a filesystem path. Use one worker for one model instance unless the host has capacity for more.

## Source map

- `src/main.jsx`: application workflow and camera lifecycle
- `src/storage.js`: local IndexedDB persistence
- `src/style.css`: responsive reference-inspired visual design
- `backend/main.py`: upload validation, decoding, CORS, API
- `backend/model.py`: development and ONNX inference adapters
- `vite.config.js`: frontend build, API proxy, PWA shell

No hosting account has been configured or deployment performed by this project.
