import httpx
import numpy as np
import pytest
from backend.gemini import GeminiAssessor, AssessmentError

def test_structured_assessment(monkeypatch):
    def post(url, **kwargs):
        assert url.startswith('https://generativelanguage.googleapis.com/')
        assert kwargs['headers']['x-goog-api-key'] == 'test-key'
        assert kwargs['json']['contents'][0]['parts'][1]['inlineData']['data']
        return httpx.Response(200,json={'candidates':[{'content':{'parts':[{'text':'{"status":"rust_detected","message":"Visible corrosion-like patches."}'}]}}]})
    monkeypatch.setattr(httpx,'post',post)
    result=GeminiAssessor('test-key').predict(np.zeros((10,10,3),dtype=np.uint8))
    assert result['status']=='rust_detected'
    assert result['confidence'] is None

@pytest.mark.parametrize('code,data', [(429,{}),(403,{}),(200,{}),(200,{'candidates':[{'content':{'parts':[{'text':'{"status":"invented","message":"test"}'}]}}]})])
def test_provider_failures_do_not_fabricate_results(monkeypatch,code,data):
    monkeypatch.setattr(httpx,'post',lambda *a,**kw:httpx.Response(code,json=data))
    with pytest.raises(AssessmentError):
        GeminiAssessor('test-key').predict(np.zeros((10,10,3),dtype=np.uint8))
