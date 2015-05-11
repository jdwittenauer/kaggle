import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sampleSubmission.csv')
training_labels = LabelEncoder().fit_transform(train['target'])

# SVMs tend to like features that look similar to ~ N(0,1), so let's stabilise the long tails
train_features = train.drop('target', axis=1)
train_features[train_features > 4] = 4

model = LinearSVC().fit(train_features, training_labels)

scores = model.decision_function(test)
predictions = 1.0 / (1.0 + np.exp(-scores))
row_sums = predictions.sum(axis=1)
predictions_normalised = predictions / row_sums[:, np.newaxis]

# create submission file
prediction_DF = pd.DataFrame(predictions_normalised, index=sample_submission.id.values,
                             columns=sample_submission.columns[1:])
prediction_DF.to_csv('svc_submission.csv', index_label='id')
