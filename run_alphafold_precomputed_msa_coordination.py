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
df.dropna(inplace=True)
df['seq_len'] = df['Amino acid sequence'].map(len)
df.sort_values(by='seq_len', inplace=True)
start_i = int(len(df)*args.start_perc/100)
end_i = int(len(df)*args.end_perc/100)
all_available_gpus = []


class GpuAllocator:

    def __init__(self):
        self.available_gpus = set(GPUtil.getAvailable(order='PCI_BUS_ID',
                                             limit=100, # big M
                                             maxLoad=0.5,
                                             maxMemory=0.5,
                                             includeNan=False, excludeID=[], excludeUUID=[]))
        self.process_id_2_gpu_id = dict()

    def check_dead_processes(self):
        for process in list(self.process_id_2_gpu_id.keys()):
            if process.poll() is not None:
                self.available_gpus.add(self.process_id_2_gpu_id[process])
                del self.process_id_2_gpu_id[process]

    def assign_process_to_gpu(self, process, gpu_id):
        self.process_id_2_gpu_id[process] = gpu_id
        self.available_gpus.remove(gpu_id)

    def is_gpu_available(self):
        return len(self.available_gpus) > 0

    def get_available_gpu(self):
        assert self.is_gpu_available(), 'No gpus'
        return list(self.available_gpus)[0]

    def wait_for_free_gpu(self):
        while not self.is_gpu_available():
            self.check_dead_processes()
            time.sleep(5)


gpu_allocator = GpuAllocator()

for _, row in df.iloc[start_i: end_i].iterrows():
    query_sequence = row['Amino acid sequence'].replace('\w', '').replace('\n', '')
    jobname = row['Uniprot ID']
    gpu_allocator.wait_for_free_gpu()
    free_gpu_id = gpu_allocator.get_available_gpu()
    try:
        open_process = subprocess.Popen(['python', '-m', 'run_alphafold_precomputed_msa',
                         '--gpu-id', str(free_gpu_id),
                         '--query-sequence', query_sequence,
                         '--jobname', jobname,
                          '--data-root', args.data_root])
    except TypeError:
        continue
    gpu_allocator.assign_process_to_gpu(open_process, free_gpu_id)