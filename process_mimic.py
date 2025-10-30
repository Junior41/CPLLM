from pyhealth.datasets import MIMIC4Dataset
from tqdm import tqdm
import pickle

# Diretório de saída
MIMIC_IV_ROOT = '../../../nfs/home/ajbrandao/cpllm/mimiciv/physionet.org/files/mimiciv/2.0/hosp/'
OUTPUT_DIR = '../../../nfs/home/ajbrandao/cpllm/mimiciv/physionet.org/files/mimiciv/2.0/outputs'

# Carregando o dataset MIMIC-IV
mimic_ds = MIMIC4Dataset(
    root=MIMIC_IV_ROOT,
    tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
    code_mapping={"ICD9CM": "CCSCM", "ICD10CM": "CCSCM", "ICD9PROC": "CCSPROC", "ICD10PROC": "CCSPROC", "NDC": "ATC"},
)

print(mimic_ds.stat())




