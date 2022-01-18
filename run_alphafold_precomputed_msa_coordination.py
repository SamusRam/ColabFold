import pandas as pd
import subprocess
import argparse
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import GPUtil
import time

parser = argparse.ArgumentParser()
parser.add_argument('--start-perc', type=float, default=0)
parser.add_argument('--end-perc', type=float, default=100)
parser.add_argument('--data-root', type=str)
args = parser.parse_args()

# tps_df = pd.read_excel(os.path.join(args.data_root, 'TPS-database_2021_11_04.xlsx'), engine='openpyxl')
tps_df = pd.read_csv(os.path.join(args.data_root, 'TPS-database_2021_11_04.csv'))
rf_df = pd.read_csv(os.path.join(args.data_root, 'tps_detection_plants_new_proteins_df.csv'))
df = pd.concat((tps_df[['Uniprot ID', 'Amino acid sequence']], rf_df[['Uniprot ID', 'Amino acid sequence']]))
df.drop_duplicates(subset=['Uniprot ID'], inplace=True)
df['seq_len'] = df['Amino acid sequence'].map(len)
df.sort_values(by='seq_len', inplace=True)
start_i = int(len(df)*args.start_perc/100)
end_i = int(len(df)*args.end_perc/100)
all_available_gpus = []


def get_free_gpu_id():
    global all_available_gpus
    if len(all_available_gpus) == 0:
        time.sleep(30)
        all_available_gpus = GPUtil.getAvailable(order='PCI_BUS_ID',
                                             limit=100, # big M
                                             maxLoad=0.5,
                                             maxMemory=0.5,
                                             includeNan=False, excludeID=[], excludeUUID=[])
        while len(all_available_gpus) == 0:
            all_available_gpus = GPUtil.getAvailable(order='PCI_BUS_ID',
                                             limit=100, # big M
                                             maxLoad=0.1,
                                             maxMemory=0.1,
                                             includeNan=False, excludeID=[], excludeUUID=[])
            time.sleep(0.5)

    return all_available_gpus.pop()


for _, row in df.iloc[start_i: end_i].iterrows():
    query_sequence = row['Amino acid sequence'].replace('\w', '').replace('\n', '')
    jobname = row['Uniprot ID']
    free_gpu_id = get_free_gpu_id()
    subprocess.Popen(['python', '-m', 'run_alphafold_precomputed_msa',
                     '--gpu-id', str(free_gpu_id),
                     '--query-sequence', query_sequence,
                     '--jobname', jobname,
                      '--data-root', args.data_root])
