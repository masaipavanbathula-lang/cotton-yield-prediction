from fastapi.testclient import TestClient
from main import app
client=TestClient(app)
def test_health(): assert client.get('/api/health').status_code==200
def test_summary():
 r=client.get('/api/dataset/summary'); assert r.status_code==200; assert r.json()['data']['records']==80000
def test_invalid_prediction(): assert client.post('/api/predict',json={}).status_code==422
def test_pagination():
 r=client.get('/api/dataset/records?page=2&page_size=10'); assert r.status_code==200; assert len(r.json()['data']['records'])==10
