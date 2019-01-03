"""
    3.1 Create train/validation/test splits

    This script will split the data/data.tsv into train/validation/test.tsv files.
"""
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


file = open("data/data.tsv", 'r')
temp = file.readlines()

subjective_feats = []
subjective_labels = []
objective_feats = []
objective_labels = []

for i in range(len(temp)):
    line = temp[i].split('\t')
    line[1].strip('\n')
    if i > 0:
        if int(line[1]) == 1:
            subjective_feats.append(line[0])
            subjective_labels.append(int(line[1]))
        elif int(line[1]) == 0:
            objective_feats.append(line[0])
            objective_labels.append(int(line[1]))

subjective_feats = np.array(subjective_feats)
subjective_labels  = np.array(subjective_labels)
objective_feats = np.array(objective_feats)
objective_labels = np.array(objective_labels)


subjective_feat_temp, test_features_subj, subjective_labels_temp, test_labels_subj = train_test_split(subjective_feats, subjective_labels, test_size=0.2,random_state=0)
train_feat_subj, valid_features_subj, train_labels_subj, valid_labels_subj = train_test_split(subjective_feat_temp, subjective_labels_temp, test_size=0.2, random_state=0)

objective_feat_temp, test_features_obj, objective_labels_temp, test_labels_obj = train_test_split(objective_feats, objective_labels, test_size=0.2, random_state=0)
train_features0, valid_features0, train_labels0, valid_labels0 = train_test_split(objective_feat_temp, objective_labels_temp, test_size=0.2, random_state=0)

train_features = np.concatenate((train_features0, train_feat_subj))
valid_features = np.concatenate((valid_features0, valid_features_subj))
test_features = np.concatenate((test_features_obj, test_features_subj))

train_labels = np.concatenate((train_labels0, train_labels_subj))
valid_labels = np.concatenate((valid_labels0, valid_labels_subj))
test_labels = np.concatenate((test_labels_obj, test_labels_subj))

trainfeats = pd.DataFrame({"text": train_features, "label": train_labels})
trainfeats.to_csv('data/train.tsv', sep='\t')
validfeats = pd.DataFrame({"text": valid_features, "label": valid_labels})
validfeats.to_csv('data/validation.tsv', sep='\t')
testfeats = pd.DataFrame({"text": test_features, "label": test_labels})
testfeats.to_csv('data/test.tsv', sep='\t')

