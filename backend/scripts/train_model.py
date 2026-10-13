import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.core.config import get_settings
from app.ml.pipeline import train
if __name__=='__main__':
    s=get_settings(); meta=train(s.data_file,s.models_path); print(f"Selected model: {meta['selected_model']}"); print(*meta['metrics'],sep='\n')
