import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import get_settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
s=get_settings(); app=FastAPI(title=s.app_name,version='1.0.0',description='Research-grounded cotton yield regression API')
app.add_middleware(CORSMiddleware,allow_origins=s.origins,allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
app.include_router(router)
@app.get('/')
def root(): return {'service':s.app_name,'docs':'/docs','health':'/api/health'}
