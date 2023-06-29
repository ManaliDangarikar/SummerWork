import numpy as np
import pandas as pd 

df = pd.read_excel('E:/CE/Summer2023/fgsm.xlsx')
data_array = df.to_numpy()

accuracy_values_clean = data_array[:, 2]      # column 2 for clean accuracy %

accuracy_mean_clean = np.mean(accuracy_values_clean, axis=0)
accuracy_std_clean = np.std(accuracy_values_clean, axis=0)
print('Accuracy Values Clean: ', accuracy_values_clean)
print('Accuracy Mean Clean:', accuracy_mean_clean)
print('Accuracy Standard Deviation Clean:', accuracy_std_clean)

accuracy_values_adversarial = data_array[:, 4]      # column 4 for adversary accuracy %

accuracy_mean_adversarial = np.mean(accuracy_values_adversarial, axis=0)
accuracy_std_adversarial = np.std(accuracy_values_adversarial, axis=0)
print('Accuracy Values Adversarial: ', accuracy_values_adversarial)
print('Accuracy Mean Adversarial:', accuracy_mean_adversarial)
print('Accuracy Standard Deviation Adversarial:', accuracy_std_adversarial)
