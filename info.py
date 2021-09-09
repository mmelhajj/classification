from pathlib import Path

root_dir = Path(__file__).parent
sar_clean_path = root_dir / 'data/clean/s1/'
sar_raw_path = root_dir / 'data/raw/s1/'
vect_clean_path = root_dir / 'data/clean/vector/'
optical_path = root_dir / 'data/clean/s2/'
outputs = root_dir / 'outputs'
otb = root_dir / 'OTB/bin/'

classfier = root_dir / 'RF_classifier/random_forest.joblib'
