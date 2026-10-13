from pathlib import Path
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from app.core.config import get_settings
from app.ml.pipeline import FEATURES, load_artifacts, train
from app.schemas.api import PredictionRequest, PredictionResponse
from app.services.data_service import DataService
import io, pandas as pd
router=APIRouter(prefix='/api')
def svc(): return DataService(get_settings().data_file)
@router.get('/health')
def health():
    s=get_settings(); _,meta=load_artifacts(s.models_path); return {'status':'ok','service':s.app_name,'model_available':meta is not None,'model_version':meta.get('model_version') if meta else None}
@router.get('/dataset/summary')
def summary(s=Depends(svc)): return {'status':'success','data':s.summary()}
@router.get('/dataset/columns')
def columns(s=Depends(svc)): return {'status':'success','data':s.columns()}
@router.get('/dataset/records')
def records(page:int=Query(1,ge=1),page_size:int=Query(20,ge=1,le=100),search:str|None=None,s=Depends(svc)): return {'status':'success','data':s.records(page,page_size,search)}
@router.get('/dataset/distribution')
def distribution(s=Depends(svc)): return {'status':'success','data':s.distribution()}
@router.get('/models/metrics')
def metrics():
    _,meta=load_artifacts(get_settings().models_path)
    return {'status':'success','data':{'metrics':meta['metrics'] if meta else [],'selected_model':meta.get('selected_model') if meta else None,'historical_paper_results':{'Linear Regression':88.721264,'Random Forest':97.886285,'SVR':92.518636,'KNN':85.088086}}}
@router.get('/models')
def models(): return metrics()
@router.get('/models/feature-importance')
def importance():
    _,meta=load_artifacts(get_settings().models_path); return {'status':'success','data':meta.get('feature_importance',{}) if meta else {}}
@router.post('/predict',response_model=PredictionResponse)
def predict(req:PredictionRequest):
    s=get_settings(); model,meta=load_artifacts(s.models_path)
    if model is None: raise HTTPException(503,'No trained model is available. Run the training command first.')
    payload=req.model_dump(); requested=payload.pop('model_name',None); x=pd.DataFrame([{k:payload.get(k) for k in FEATURES}]); pred=float(model.predict(x)[0])
    return PredictionResponse(predicted_yield=pred,model_name=meta['selected_model'],model_version=meta['model_version'],status='success',input_features=FEATURES)
@router.post('/models/train')
def train_models(sample_size:int|None=Query(None,ge=1000,le=80000)):
    s=get_settings(); return {'status':'success','data':train(s.data_file,s.models_path,sample_size)}
@router.get('/reports/evaluation')
def report():
    _,meta=load_artifacts(get_settings().models_path)
    if not meta: raise HTTPException(404,'Evaluation report is not available. Train a model first.')
    return {'status':'success','data':meta}
@router.post('/dataset/upload')
async def upload(file:UploadFile=File(...)):
    if not file.filename or not file.filename.lower().endswith('.csv'): raise HTTPException(400,'Only CSV files are accepted.')
    raw=await file.read()
    if len(raw)>25*1024*1024: raise HTTPException(413,'File exceeds the 25 MB upload limit.')
    try: d=pd.read_csv(io.BytesIO(raw))
    except Exception as e: raise HTTPException(400,f'Invalid CSV: {e}')
    missing=[c for c in FEATURES+['Yield'] if c not in d.columns]
    if missing: raise HTTPException(400,{'missing_columns':missing})
    return {'status':'success','data':{'filename':file.filename,'rows':len(d),'columns':len(d.columns),'message':'Validated successfully; original dataset was not replaced.'}}
