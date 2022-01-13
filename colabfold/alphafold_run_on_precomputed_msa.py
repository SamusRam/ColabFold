import re
import os
import argparse
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'


parser = argparse.ArgumentParser()
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--num-models', type=int, default=5)
parser.add_argument('--msa-mode', default='"MMseqs2 (UniRef+Environmental)"',
                    type=str,
                    help='"MMseqs2 (UniRef+Environmental)", "MMseqs2 (UniRef only)","single_sequence","custom"')
parser.add_argument('--num-recycles', type=int, default=3)
parser.add_argument('--use-templates', type=bool, default=True)
parser.add_argument('--query-sequence', type=str)
parser.add_argument('--result-dir', type=str, default='.')
parser.add_argument('--jobname', type=str)


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

from colabfold.utils import setup_logging
from colabfold.batch import get_queries, run_on_precomputed_msa
import argparse
import hashlib
from pathlib import Path
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


def add_hash(x,y):
  return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]

msa_mode = args.msa_mode
num_models = args.num_models
num_recycles = args.num_recycles
use_msa = True if "MMseqs2" in msa_mode else False
use_env = True if msa_mode == "MMseqs2 (UniRef+Environmental)" else False
use_amber = True
use_templates = args.use_templates

query_sequence = args.query_sequence
query_sequence = "".join(query_sequence.split())

jobname = args.jobname
jobname = "".join(jobname.split())
jobname = re.sub(r'\W+', '', jobname)
jobname = add_hash(jobname, query_sequence)

with open(f"{jobname}.csv", "w") as text_file:
    text_file.write(f"id,sequence\n{jobname},{query_sequence}")

queries_path=f"{jobname}.csv"

with open(f"{jobname}.log", "w") as text_file:
    text_file.write("num_models=%s\n" % num_models)
    text_file.write("use_amber=%s\n" % use_amber)
    text_file.write("use_msa=%s\n" % use_msa)
    text_file.write("msa_mode=%s\n" % msa_mode)
    text_file.write("use_templates=%s\n" % use_templates)


result_dir = args.result_dir
setup_logging(Path(".").joinpath("log.txt"))
queries, is_complex = get_queries(queries_path)
run_on_precomputed_msa(
    queries=queries,
    result_dir=result_dir,
    use_templates=use_templates,
    use_amber=use_amber,
    msa_mode=msa_mode,
    num_models=num_models,
    num_recycles=num_recycles,
    model_order=[1, 2, 3, 4, 5],
    is_complex=is_complex,
    data_dir=Path("."),
    keep_existing_results=False,
    recompile_padding=1.0,
    rank_mode="auto",
    pair_mode="unpaired+paired",
    stop_at_score=float(100)
)