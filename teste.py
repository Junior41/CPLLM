
import numpy as np
import pickle
train_pickle_file_path = '../../../nfs/home/ajbrandao/cpllm/mimiciv/multitask/chronic_kidney_disease_descriptions_train.pickle'

# Load the pickle file
def load_pickle_file(pickle_file_path):
    with open(pickle_file_path, "rb") as file:
        data = pickle.load(file)
    return data

class MedicalDataset():
    def __init__(self, pickle_file_path):
        all_data = load_pickle_file(pickle_file_path)

        patient_ids = []
        visit_ids = []
        labels_readmission = []
        labels_disease_prediction = []
        diagnoses = []

        for data in all_data:
            patient_ids.append(data[0])
            labels_disease_prediction.append(data[1][0])
            labels_readmission.append(data[1][1])
            diagnoses_list = data[2]
            diagnoses.append(diagnoses_list)
            visit_ids.append(data[3])
            
        combined_labels = np.array(list(zip(labels_disease_prediction, labels_readmission)))
        
        self.data_dict = {
            'visit_id': visit_ids,
            'patient_id': patient_ids,
            'diagnoses': diagnoses,
            'labels': combined_labels.tolist() 
        }

    def to_dict(self):
        return self.data_dict


train_dataset = MedicalDataset(train_pickle_file_path)

train_dict = train_dataset.to_dict()

first_row = {
    "visit_id": train_dict["visit_id"][0],
    "patient_id": train_dict["patient_id"][0],
    "diagnoses": train_dict["diagnoses"][0],
    "labels": train_dict["labels"][0]
}

print(first_row)
