from importlib import resources as impresources
from pathlib import Path
from classes.data import Data
from classes.annotation import Annotation

# Path to dataset
data_path = Path('') #Enter dataset path here

# Build recordings list
sub_list = [x for x in data_path.glob("sub*")]

# Only include first 10 subjects (sub-001 to sub-010)
sub_list = [x for x in sub_list if int(x.name.split('-')[-1]) <= 10]

recordings = [
    [x.name, xx.name.split('_')[-2]]
    for x in sub_list
    for xx in (x / 'ses-01' / 'mov').glob("*edf")
]

data = []
annotations = []

for rec in recordings:
    print(rec[0] + ' ' + rec[1])
    rec_data = Data.loadData(data_path.as_posix(), rec, modalities=['mov'])
    rec_annotations = Annotation.loadAnnotation(data_path.as_posix(), rec)

    data.append(rec_data)
    annotations.append(rec_annotations)

