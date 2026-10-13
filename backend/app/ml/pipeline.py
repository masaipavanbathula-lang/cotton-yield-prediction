from __future__ import annotations
import json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

FEATURES = ['Growth_Cycle','Sunlight_Hours','Precipitation','Average_Temperature','Drought_Days','Flood_Days','Soil_Moisture','Soil_pH','CO2_Concentration','Sunlight_Level','Flood_Level','Drought_Level']
NUMERIC = FEATURES[:9]
CATEGORICAL = FEATURES[9:]
MODEL_NAMES = ['Linear Regression','Random Forest','SVR','KNN','Gradient Boosting','XGBoost','LightGBM']

def load_data(path): return pd.read_csv(path)
def make_preprocessor():
    return ColumnTransformer([('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())]), NUMERIC), ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL)])
def available_models():
    models = {'Linear Regression': LinearRegression(), 'Random Forest': RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1, max_depth=18), 'SVR': SVR(kernel='rbf', C=10), 'KNN': KNeighborsRegressor(n_neighbors=7, weights='distance'), 'Gradient Boosting': GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=3, random_state=42)}
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = XGBRegressor(n_estimators=160, learning_rate=0.05, max_depth=6, subsample=.8, colsample_bytree=.8, random_state=42, n_jobs=-1, objective='reg:squarederror')
    except ImportError: pass
    try:
        from lightgbm import LGBMRegressor
        models['LightGBM'] = LGBMRegressor(n_estimators=160, learning_rate=.05, num_leaves=31, random_state=42, verbosity=-1)
    except ImportError: pass
    return models

def train(data_path, model_dir, sample_size=None):
    started=time.time(); df=load_data(data_path)
    if sample_size and len(df)>sample_size: df=df.sample(sample_size, random_state=42)
    X=df[FEATURES].copy(); y=df['Yield'].astype(float)
    X_train, X_temp, y_train, y_temp=train_test_split(X,y,test_size=.30,random_state=42)
    X_val, X_test, y_val, y_test=train_test_split(X_temp,y_temp,test_size=.50,random_state=42)
    models={}; metrics=[]; predictions={}; importances={}
    for name, estimator in available_models().items():
        pipe=Pipeline([('preprocessor',make_preprocessor()),('model',estimator)])
        t=time.time(); pipe.fit(X_train,y_train); pred=pipe.predict(X_test)
        metrics.append({'model_name':name,'mae':float(mean_absolute_error(y_test,pred)),'rmse':float(mean_squared_error(y_test,pred)**.5),'r2':float(r2_score(y_test,pred)),'training_seconds':round(time.time()-t,3),'test_samples':len(y_test)})
        predictions[name]={'actual':y_test.tolist()[:500], 'predicted':pred.tolist()[:500]}
        if hasattr(estimator,'feature_importances_'):
            names=list(pipe.named_steps['preprocessor'].get_feature_names_out()); vals=estimator.feature_importances_; importances[name]=sorted([{'feature':n.replace('numeric__','').replace('categorical__',''),'importance':float(v)} for n,v in zip(names,vals)], key=lambda z:z['importance'], reverse=True)[:15]
        models[name]=pipe
    metrics.sort(key=lambda x:x['r2'], reverse=True); best=metrics[0]['model_name']
    out=Path(model_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[best], out/'selected_model.joblib'); joblib.dump(models, out/'models.joblib')
    meta={'model_version':'1.0.0','selected_model':best,'features':FEATURES,'numeric_features':NUMERIC,'categorical_features':CATEGORICAL,'metrics':metrics,'predictions':predictions,'feature_importance':importances,'trained_at':pd.Timestamp.utcnow().isoformat(),'rows_used':len(df),'split':{'train':len(y_train),'validation':len(y_val),'test':len(y_test)},'duration_seconds':round(time.time()-started,2)}
    (out/'metadata.json').write_text(json.dumps(meta, indent=2))
    return meta

def load_artifacts(model_dir):
    p=Path(model_dir); meta=p/'metadata.json'; model=p/'selected_model.joblib'
    if not meta.exists() or not model.exists(): return None, None
    return joblib.load(model), json.loads(meta.read_text())
