import io
from PIL import Image
from fastapi.testclient import TestClient
from backend.main import app

def test_development_pipeline(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('RUST_MODEL_PATH', raising=False)
    image=io.BytesIO(); Image.new('RGB',(20,20),(170,80,30)).save(image,format='PNG')
    with TestClient(app) as client:
        response=client.post('/api/detect-rust',files={'file':('metal.png',image.getvalue(),'image/png')})
        assert response.status_code==200
        result=response.json()
        assert result['status']=='unable_to_determine'
        assert result['mode']=='development'
        assert result['confidence'] is None
        assert result['rust_coverage_percent'] is None

def test_rejects_bad_uploads(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('RUST_MODEL_PATH', raising=False)
    with TestClient(app) as client:
        for content, mime, status in [(b'not an image','image/png',415),(b'','image/png',400),(b'text','text/plain',415),(b'x'*(10*1024*1024+1),'image/jpeg',413)]:
            assert client.post('/api/detect-rust',files={'file':('test',content,mime)}).status_code==status
