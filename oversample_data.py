# %% Load raw dataset
import pandas as pd

data = pd.read_csv("data/raw/HIV_train.csv")

# %% Apply oversampling

print('Before Oversampling:')
print(data.shape)
print(data["HIV_active"].value_counts())
print("\n")

class_0 = data[data['HIV_active'] == 0].sample(35850, replace=True)
class_1 = data[data['HIV_active'] == 1].sample(35850, replace=True)
data = pd.concat([class_0, class_1], axis=0)

print('After Oversampling:')
print(data.shape)
print(data["HIV_active"].value_counts())

# %% Save
data.to_csv("data/raw/HIV_train_oversampled.csv")