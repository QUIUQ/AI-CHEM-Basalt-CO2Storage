import os
import pandas as pd
from tqdm import tqdm

base_dir = r'D:\toughreact-run'

# Get all the files
case_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

case_numbers = []

for case_dir in tqdm(case_dirs, desc="Checking cases"):
    file_path = os.path.join(case_dir, 'react.csv')
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            if 'TIME' in df.columns and 31536000. not in df['TIME'].unique():
                case_number = int(case_dir.split('_')[-1])
                case_numbers.append(case_number)
        except Exception as e:
            pass

case_numbers.sort()

with open('errorcase.txt', 'w') as f:
    for case_number in case_numbers:
        f.write(f'{case_number}\n')