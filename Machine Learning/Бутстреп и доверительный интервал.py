import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 10, 6


# Generate subsamples
def get_bootstrap_samples(data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples


# Interval evaluation
def stat_intervals(stat, alpha):
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries


telecom_data = pd.read_csv('data/telecom_churn.csv')

fig = sns.kdeplot(telecom_data[telecom_data['Churn'] == False]['Customer service calls'], label='Loyal')
fig = sns.kdeplot(telecom_data[telecom_data['Churn'] == True]['Customer service calls'], label='Churn')
fig.set(xlabel='Количество звонков', ylabel='Плотность')
plt.show()

loyal_calls = telecom_data[telecom_data['Churn'] == False]['Customer service calls'].values
churn_calls = telecom_data[telecom_data['Churn'] == True]['Customer service calls'].values

np.random.seed(0)

loyal_mean_scores = [np.mean(sample) for sample in get_bootstrap_samples(loyal_calls, 1_000)]
print(len(loyal_mean_scores))
churn_mean_scores = [np.mean(sample) for sample in get_bootstrap_samples(churn_calls, 1_000)]

print('Service calls from loyal: mean interval ', stat_intervals(loyal_mean_scores, 0.05))
print('Service calls from churn: mean interval ', stat_intervals(churn_mean_scores, 0.05))
