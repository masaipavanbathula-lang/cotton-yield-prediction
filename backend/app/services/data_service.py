import pandas as pd
from pathlib import Path
from app.ml.pipeline import FEATURES
class DataService:
    def __init__(self,path): self.path=Path(path); self._df=None
    @property
    def df(self):
        if self._df is None: self._df=pd.read_csv(self.path)
        return self._df
    def summary(self):
        d=self.df; y=d['Yield']; return {'records':len(d),'columns':len(d.columns),'input_features':len(FEATURES),'features':FEATURES,'target':'Yield','yield':{'mean':float(y.mean()),'min':float(y.min()),'max':float(y.max()),'median':float(y.median())},'missing_values':{k:int(v) for k,v in d.isna().sum().items()},'duplicates':int(d.duplicated().sum()),'date_range':{'planting': [str(d.Planting_Date.min()),str(d.Planting_Date.max())], 'harvest':[str(d.Harvest_Date.min()),str(d.Harvest_Date.max())]}}
    def columns(self):
        d=self.df; return [{'name':c,'dtype':str(d[c].dtype),'missing':int(d[c].isna().sum()),'unique':int(d[c].nunique())} for c in d.columns]
    def distribution(self):
        d=self.df; bins=pd.cut(d['Yield'], bins=10); hist=d.groupby(bins, observed=False).size(); return {'yield_histogram':[{'label':str(k),'count':int(v)} for k,v in hist.items()],'growth_cycle':d['Growth_Cycle'].value_counts().sort_index().head(30).rename_axis('value').reset_index(name='count').to_dict('records'),'levels':{c:d[c].fillna('Missing').value_counts().rename_axis('value').reset_index(name='count').to_dict('records') for c in ['Sunlight_Level','Flood_Level','Drought_Level']}}
    def records(self,page=1,page_size=20,search=None):
        d=self.df
        if search: d=d[d.astype(str).apply(lambda row: row.str.contains(search,case=False,na=False).any(),axis=1)]
        total=len(d); rows=d.iloc[(page-1)*page_size:page*page_size].where(pd.notna(d),None).to_dict('records'); return {'records':rows,'page':page,'page_size':page_size,'total':total,'pages':(total+page_size-1)//page_size}
