import pandas as pd
import subprocess
import argparse
import os
import GPUtil
import time

parser = argparse.ArgumentParser()
parser.add_argument('--start-perc', type=float, default=0)
parser.add_argument('--end-perc', type=float, default=100)
parser.add_argument('--data-root', type=str)
args = parser.parse_args()

# tps_df = pd.read_excel(os.path.join(args.data_root, 'TPS-database_2021_11_04.xlsx'), engine='openpyxl')
tps_df = pd.read_excel(os.path.join(args.data_root, 'TPS-database_2021_11_04.csv'))
rf_df = pd.read_csv(os.path.join(args.data_root, 'tps_detection_plants_new_proteins_df.csv'))
df = pd.concat((tps_df[['Uniprot ID', 'Amino acid sequence']], rf_df[['Uniprot ID', 'Amino acid sequence']]))
df.drop_duplicates(subset=['Uniprot ID'], inplace=True)
df['seq_len'] = df['Amino acid sequence'].map(len)
df.sort_values(by='seq_len', inplace=True)
start_i = int(len(df)*args.start_perc/100)
end_i = int(len(df)*args.end_perc/100)


def get_free_gpu_id():
    available_gpus = GPUtil.getAvailable(order='PCI_BUS_ID',
                                         limit=100, # big M
                                         maxLoad=0.5,
                                         maxMemory=0.5,
                                         includeNan=False, excludeID=[], excludeUUID=[])
    while len(available_gpus) == 0:
        time.sleep(2)
        available_gpus = GPUtil.getAvailable(order='PCI_BUS_ID',
                                         limit=100, # big M
                                         maxLoad=0.5,
                                         maxMemory=0.2,
                                         includeNan=False, excludeID=[], excludeUUID=[])
    return available_gpus[0]


for _, row in df.iloc[start_i: end_i].iterrows():
    query_sequence = row['Amino acid sequence'].replace('\w', '').replace('\n', '')
    jobname = row['Uniprot ID']
    free_gpu_id = get_free_gpu_id()
    subprocess.Popen(['python', '-m', 'colabfold.alphafold_run_on_precomputed_msa',
                     '--gpu-id', str(free_gpu_id),
                     '--query-sequence', query_sequence,
                     '--jobname', jobname,
                      '--data-root', args.data_root])


# conda create -n alphafold_extraction -c conda-forge -c bioconda kalign3=3.2.2 hhsuite=3.3.0 openmm=7.5.1 pdbfixer python=3.8 pandas numpy tensorflow-gpu
# source activate alphafold_extraction
# pip install --no-warn-conflicts -q "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold"
# pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.
# wget -nc https://raw.githubusercontent.com/SamusRam/ColabFold/high_quality_representations/beta/modules.patch
# python_executable_path=$(which python)
# alphafold_modules_path=${python_executable_path/bin\/python/"lib/python3.8/site-packages/alphafold/model/modules.py"}
# patch -u $alphafold_modules_path -i modules.patch

# # pip install --upgrade --force-reinstall "colabfold[alphafold] @ git+https://github.com/SamusRam/ColabFold@high_quality_representations"
# # conda install -y -c conda-forge -c bioconda kalign3=3.2.2 hhsuite=3.3.0
# # conda install -y -c conda-forge openmm=7.5.1 pdbfixer
# git clone https://github.com/SamusRam/ColabFold
# cd ColabFold
# git checkout high_quality_representations
# pip install openpyxl
# pip install GPUtil
#
# pip uninstall -y pandas
# pip install pandas

# python -m colabfold.alphafold_run_on_precomputed_msa_coordination --data-root ../data --start-perc 0 --end-perc 5

