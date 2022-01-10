import pandas as pd
import subprocess
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--start-perc', type=float, default=0)
parser.add_argument('--end-perc', type=float, default=100)
args = parser.parse_args()

tps_df = pd.read_excel('data/TPS-database_2021_11_04.xlsx', engine='openpyxl')
rf_df = pd.read_excel('data/tps_detection_plants_new_proteins_df.csv')
df = pd.concat((tps_df[['Uniprot ID', 'Amino acid sequence']], rf_df[['Uniprot ID', 'Amino acid sequence']]))
df = df.sort_values(by='Uniprot ID')
start_i = int(len(df)*args.start_perc/100)
end_i = int(len(df)*args.end_perc/100)


for _, row in df.iloc[start_i: end_i].iterrows():
    query_sequence = row['Amino acid sequence']
    jobname = row['Uniprot ID']
    subprocess.call(['python', 'msa_precomputation.py', '--max-msa-depth', '100000',
                     '--query-sequence', query_sequence, '--jobname', jobname])


# pip install "colabfold[alphafold] @ git+https://github.com/SamusRam/ColabFold@high_quality_representations"