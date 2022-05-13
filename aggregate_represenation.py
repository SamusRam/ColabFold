import numpy as np
from pathlib import Path
import pickle
from shutil import copyfile
from tqdm.auto import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start-perc', type=float, default=0)
parser.add_argument('--end-perc', type=float, default=100)
args = parser.parse_args()


result_dir = Path("alphafold_results")

selected_results_path = Path('/home/samusram/alphafold_protein_embeddings')

ready_ids = sorted(list({path.name.split('_')[0] for path in result_dir.glob(f'*_representations.pkl')}))
ready_ids = ready_ids[290:]

id_2_top_model = dict()

start_i = int(len(ready_ids)*args.start_perc/100)
end_i = int(len(ready_ids)*args.end_perc/100)

for id in tqdm(ready_ids[start_i: end_i]):
    # selecting top model
    prediction_names_splitted = [path.name.replace('.pdb', '').split('_') for path in
                                 result_dir.glob(f'{id}_*_unrelaxed_model*')]
    model_rank_2_name = {int(elements[-1]): f'model_{elements[-3]}' for elements in prediction_names_splitted}
    id_2_top_model[id] = model_rank_2_name[1]

    repr_files = list(result_dir.glob(f'{id}*_representations.pkl'))
    assert len(repr_files) >= 1, f'repr files issue with {id}'
    repr_file = repr_files[0]
    with open(repr_file, 'rb') as f:
        repr = pickle.load(f)
    repr_top_model = repr[id_2_top_model[id]]

    with open(selected_results_path/f'{id}_msa_first_row.pkl', 'wb') as f:
        pickle.dump(repr_top_model['msa_first_row'].mean(axis=0), f)

    with open(selected_results_path/f'{id}_msa.pkl', 'wb') as f:
        pickle.dump(repr_top_model['msa'].mean(axis=0).mean(axis=0), f)

    with open(selected_results_path/f'{id}_pair.pkl', 'wb') as f:
        pickle.dump(repr_top_model['pair'].mean(axis=(0, 1)), f)

    with open(selected_results_path/f'{id}_single.pkl', 'wb') as f:
        pickle.dump(repr_top_model['single'].mean(axis=0), f)

    with open(selected_results_path/f'{id}_structure_module.pkl', 'wb') as f:
        pickle.dump(repr_top_model['structure_module'].mean(axis=0), f)