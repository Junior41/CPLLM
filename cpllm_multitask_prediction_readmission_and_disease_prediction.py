import pickle
from random import randint

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, auc, precision_recall_curve, roc_auc_score
from transformers import TrainingArguments, AutoConfig, \
    AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding, Trainer
from transformers.integrations import WandbCallback
import numpy as np
from datasets import Dataset
import wandb
from torch.nn import CrossEntropyLoss

# --- Configurações Multitask ---
NUM_LABELS = 4 
EPOCHS = 4
MAX_LENGTH = 2048
# Ajuste o OUTPUT_DIR para o novo nome de pasta (consistente com a preparação de dados)
OUTPUT_DIR = "../../../nfs/home/ajbrandao/cpllm/outputMultiTaskMIMICReadmissionAnddiseasePredictionWithDrugs"
MODEL_ID = "meta-llama/Llama-2-13b-hf"

# DATA
# Os caminhos dos arquivos devem ser ajustados para os arquivos gerados na etapa anterior
train_pickle_file_path = '../../../nfs/home/ajbrandao/cpllm/mimiciv/multitaskWithDrugs/chronic_kidney_disease_descriptions_train.pickle'
validation_pickle_file_path = '../../../nfs/home/ajbrandao/cpllm/mimiciv/multitaskWithDrugs/chronic_kidney_disease_descriptions_validation.pickle'
test_pickle_file_path = '../../../nfs/home/ajbrandao/cpllm/mimiciv/multitaskWithDrugs/chronic_kidney_disease_descriptions_test.pickle'

# Configuração para Quantização em 4 bits
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

# O modelo agora é inicializado com 4 rótulos de saída
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    config=AutoConfig.from_pretrained(MODEL_ID,
                                      trust_remote_code=True,
                                      num_labels=NUM_LABELS),
    trust_remote_code=True,
    quantization_config=bnb_config
)

print(model)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
print_trainable_parameters(model)
model = get_peft_model(model, config)
print_trainable_parameters(model)

model.config.pad_token_id = model.config.eos_token_id
model.resize_token_embeddings(len(tokenizer))


# --- Função de Métrica Multitask ---
def compute_metrics(p):
    # p.predictions tem shape (batch_size, 4)
    # p.label_ids tem shape (batch_size, 2)

    # 1. Separa as predições e rótulos por tarefa
    # Predições: [logits_disease_prediction_0, logits_disease_prediction_1, logits_readmission_0, logits_readmission_1]
    logits_disease_prediction = p.predictions[:, 0:2]
    logits_readmission = p.predictions[:, 2:4]
    
    # Rótulos: [label_disease_prediction, label_readmission]
    labels_disease_prediction = p.label_ids[:, 0]
    labels_readmission = p.label_ids[:, 1]

    # 2. Converte logits para probabilidades e predições de classe
    probs_disease_prediction = torch.softmax(torch.from_numpy(logits_disease_prediction), dim=1).numpy()
    probs_readmission = torch.softmax(torch.from_numpy(logits_readmission), dim=1).numpy()
    
    preds_disease_prediction = np.argmax(probs_disease_prediction, axis=1)
    preds_readmission = np.argmax(probs_readmission, axis=1)
   
    # 4. Calcula as métricas para disease_prediction
    acc_disease_prediction = accuracy_score(labels_disease_prediction, preds_disease_prediction)
    prec_rec_f1_disease_prediction = precision_recall_fscore_support(labels_disease_prediction, preds_disease_prediction, average='binary', zero_division=0)
    aucpr_disease_prediction = auc(precision_recall_curve(labels_disease_prediction, probs_disease_prediction[:, 1], pos_label=1)[1], 
                            precision_recall_curve(labels_disease_prediction, probs_disease_prediction[:, 1], pos_label=1)[0])
    auroc_disease_prediction = roc_auc_score(labels_disease_prediction, probs_disease_prediction[:, 1])

    # 3. Calcula as métricas para Readmissão
    acc_readmission = accuracy_score(labels_readmission, preds_readmission)
    prec_rec_f1_readmission = precision_recall_fscore_support(labels_readmission, preds_readmission, average='binary', zero_division=0)
    aucpr_readmission = auc(precision_recall_curve(labels_readmission, probs_readmission[:, 1], pos_label=1)[1], 
                          precision_recall_curve(labels_readmission, probs_readmission[:, 1], pos_label=1)[0])
    auroc_readmission = roc_auc_score(labels_readmission, probs_readmission[:, 1])

    # 5. Combina as métricas
    return {
        'disease_prediction_accuracy': acc_disease_prediction,
        'disease_prediction_precision': prec_rec_f1_disease_prediction[0],
        'disease_prediction_recall': prec_rec_f1_disease_prediction[1],
        'disease_prediction_f1': prec_rec_f1_disease_prediction[2],
        'disease_prediction_PR-AUC': aucpr_disease_prediction,
        'isease_prediction_ROC-AUC': auroc_disease_prediction,
        
        'readmission_accuracy': acc_readmission,
        'readmission_precision': prec_rec_f1_readmission[0],
        'readmission_recall': prec_rec_f1_readmission[1],
        'readmission_f1': prec_rec_f1_readmission[2],
        'readmission_PR-AUC': aucpr_readmission,
        'readmission_ROC-AUC': auroc_readmission,
        
        # Métrica de seleção do melhor modelo (média das AUC-PR)
        'eval_multitask_aucpr': (aucpr_disease_prediction + aucpr_readmission) / 2.0,
    }



class MultitaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        current_device = logits.device 
        
        # Pesos baseados na sua distribuição:
        # Doença (1ª posição): ~24% pos -> peso 3.15
        # Readmissão (2ª posição): ~15% pos -> peso 5.63
        weight_disease = torch.tensor([1.0, 3.15]).to(current_device)
        weight_readmission = torch.tensor([1.0, 5.63]).to(current_device)
        
        # SEPARAÇÃO CORRIGIDA:
        # Se label = [doença, readmissão], então:
        # Logits: colunas 0,1 = doença | colunas 2,3 = readmissão
        logits_disease = logits[:, 0:2]
        logits_readmission = logits[:, 2:4]
        
        # Labels: coluna 0 = doença | coluna 1 = readmissão
        labels_disease = labels[:, 0].long()
        labels_readmission = labels[:, 1].long()
        
        # Funções de perda com os pesos corretos para cada tarefa
        loss_fct_d = CrossEntropyLoss(weight=weight_disease)
        loss_fct_r = CrossEntropyLoss(weight=weight_readmission)
        
        loss_d = loss_fct_d(logits_disease, labels_disease)
        loss_r = loss_fct_r(logits_readmission, labels_readmission)
        
        # Ponderação: Mantemos o peso maior (2.0) na readmissão porque ela é a mais difícil
        loss = (1.0 * loss_d) + (2.0 * loss_r)
        
        return (loss, outputs) if return_outputs else loss


# Load the pickle file
def load_pickle_file(pickle_file_path):
    with open(pickle_file_path, "rb") as file:
        data = pickle.load(file)
    return data


# --- Custom Dataset para Multitask ---
class MedicalDataset():
    def __init__(self, pickle_file_path):
        all_data = load_pickle_file(pickle_file_path)

        patient_ids = []
        visit_ids = []
        labels_readmission = []
        labels_disease_prediction = []
        diagnoses = []
        drugs = []
        procedures = []

        for data in all_data:
            patient_ids.append(data[0])
            labels_disease_prediction.append(data[1][0])
            labels_readmission.append(data[1][1])
            
            # Função auxiliar interna para garantir que listas virem strings
            def to_str(item):
                if isinstance(item, list):
                    return ", ".join([str(i) for i in item])
                return str(item) if item is not None else ""

            # Transformando tudo em string antes de adicionar às listas
            diagnoses.append(to_str(data[2]))
            visit_ids.append(data[3])
            drugs.append(to_str(data[4]))
            procedures.append(to_str(data[6]))
            
        combined_labels = np.array(list(zip(labels_disease_prediction, labels_readmission)))
        
        self.data_dict = {
            'visit_id': visit_ids,
            'patient_id': patient_ids,
            'diagnoses': diagnoses,
            'drugs': drugs,
            'procedures': procedures,
            'labels': combined_labels.tolist() 
        }

    def to_dict(self):
        return self.data_dict


train_dataset = MedicalDataset(train_pickle_file_path)
test_dataset = MedicalDataset(test_pickle_file_path)
validation_dataset = MedicalDataset(validation_pickle_file_path)

# Convert custom datasets to dictionaries
train_dict = train_dataset.to_dict()
test_dict = test_dataset.to_dict()
validation_dict = validation_dataset.to_dict()

# Convert dictionaries to datasets
train_dataset = Dataset.from_dict(train_dict)
test_dataset = Dataset.from_dict(test_dict)
validation_dataset = Dataset.from_dict(validation_dict)

prompt_template = """
Your task is to perform a multitask prediction based on the patient's diagnosis, procedures, and drugs provided below.
You must output two labels:

1. **Disease label**: Determine whether the patient is likely to have Chronic Kidney Disease.
2. **Readmission label**: Determine whether the patient is likely to be readmitted to the hospital.

Each description is separated by a comma.

**Patient Diagnosis Descriptions:**

{diagnoses}

**Patient Drugs Descriptions:**

{drugs}

**Patient Procedures Descriptions:**

{procedures}
"""



def template_dataset(sample):
    sample["text"] = prompt_template.format(diagnoses=sample["diagnoses"],
                                            drugs=sample["drugs"],
                                            procedures=sample["procedures"],
                                            eos_token=tokenizer.eos_token)
    return sample


# apply prompt template per sample
train_dataset = train_dataset.map(template_dataset)
print("\n" + "="*100)
print("VISUALIZAÇÃO DOS 5 PRIMEIROS PROMPTS (CAMPO 'TEXT'):")
print("="*100)

for i in range(5):
    print(f"\n--- EXEMPLO {i+1} ---")
    print(train_dataset[i]["text"])
    print("-" * 50)

print("="*100 + "\n")
validation_dataset = validation_dataset.map(template_dataset)
test_dataset = test_dataset.map(template_dataset)

print(train_dataset[randint(0, len(train_dataset))])

# 2. Aplica o tokenizador (que agora pode acessar o campo 'text')
lm_train_dataset = train_dataset.map(
    lambda sample: {
        "input_ids": tokenizer(sample["text"], truncation=True, max_length=MAX_LENGTH).input_ids,
        "attention_mask": tokenizer(sample["text"], truncation=True, max_length=MAX_LENGTH).attention_mask,
        "labels": sample["labels"]
    },
    batched=True,
    batch_size=256,
    num_proc=128,
    remove_columns=list(train_dataset.features)
)
lm_validation_dataset = validation_dataset.map(
    lambda sample: {
        "input_ids": tokenizer(sample["text"], truncation=True, max_length=MAX_LENGTH).input_ids,
        "attention_mask": tokenizer(sample["text"], truncation=True, max_length=MAX_LENGTH).attention_mask,
        "labels": sample["labels"] # Usa a coluna 'labels' combinada
    },
    batched=True,
    batch_size=256,
    num_proc=128,
    remove_columns=list(validation_dataset.features)
)
lm_test_dataset = test_dataset.map(
    lambda sample: {
        "input_ids": tokenizer(sample["text"], truncation=True, max_length=MAX_LENGTH).input_ids,
        "attention_mask": tokenizer(sample["text"], truncation=True, max_length=MAX_LENGTH).attention_mask,
        "labels": sample["labels"] # Usa a coluna 'labels' combinada
    },
    batched=True,
    batch_size=256,
    num_proc=128,
    remove_columns=list(test_dataset.features)
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(validation_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy='steps',
    save_strategy='steps',
    eval_steps=1000,
    save_steps=1000,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    auto_find_batch_size=True,
    logging_steps=10,
    learning_rate=2e-5,
    optim="adamw_torch",
    save_total_limit=20,
    logging_dir='./logs',
    load_best_model_at_end=True,
    # A métrica de seleção agora é a média das AUC-PR das duas tarefas
    metric_for_best_model="eval_multitask_aucpr", 
    greater_is_better=True,
    dataloader_num_workers=8,
)

wandb.init(
    project="CPLLM",
    name="experimento-multitask-disease-readmission", # Nome do run ajustado
    config={
        "epochs": EPOCHS,
        "model": MODEL_ID,
        "max_length": MAX_LENGTH,
        "batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "tasks": "Disease and Readmission" # Nova informação
    }
)

model.config.use_cache = False
# Usamos o MultitaskTrainer customizado
trainer = MultitaskTrainer( 
    model=model,
    args=training_args,
    train_dataset=lm_train_dataset,
    eval_dataset=lm_validation_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    callbacks=[WandbCallback]
)

trainer.train()
trainer.save_model(OUTPUT_DIR)

test_results = trainer.evaluate(eval_dataset=lm_test_dataset)

wandb.log(test_results)

print(f'test_results= {test_results}')
print(f'see outputs in= {OUTPUT_DIR}')
trainer.save_metrics("test", test_results)
