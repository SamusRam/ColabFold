import pandas as pd
import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--start-perc', type=float, default=0)
parser.add_argument('--end-perc', type=float, default=100)
parser.add_argument('--data-root', type=str)
args = parser.parse_args()

tps_df = pd.read_excel(os.path.join(args.data_root, 'TPS-database_2021_11_04.xlsx'), engine='openpyxl')
rf_df = pd.read_csv(os.path.join(args.data_root, 'tps_detection_plants_new_proteins_df.csv'))
df = pd.concat((tps_df[['Uniprot ID', 'Amino acid sequence']], rf_df[['Uniprot ID', 'Amino acid sequence']]))
df.drop_duplicates(subset=['Uniprot ID'], inplace=True)
df.sort_values(by='Uniprot ID', inplace=True)
start_i = int(len(df)*args.start_perc/100)
end_i = int(len(df)*args.end_perc/100)


for _, row in df.iloc[start_i: end_i].iterrows():
    query_sequence = row['Amino acid sequence'].replace('\w', '').replace('\n', '')
    jobname = row['Uniprot ID']
    subprocess.call(['python', '-m', 'colabfold.msa_precomputation', '--max-msa-depth', '100000',
                     '--query-sequence', query_sequence, '--jobname', jobname])


# conda create -n msa_extraction python=3.8 pandas numpy
# conda activate msa_extraction
# pip install opencv-python
# pip install "colabfold[alphafold] @ git+https://github.com/SamusRam/ColabFold@high_quality_representations"
# pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.
# # pip install --upgrade --force-reinstall "colabfold[alphafold] @ git+https://github.com/SamusRam/ColabFold@high_quality_representations"
# conda install -y -c conda-forge -c bioconda kalign3=3.2.2 hhsuite=3.3.0
# conda install -y -c conda-forge openmm=7.5.1 pdbfixer
# git clone https://github.com/SamusRam/ColabFold
# git checkout high_quality_representations
# pip install openpyxl
# python -m colabfold.msa_precomputation_coordination --data-root ../data --start-perc 20 --end-perc 40