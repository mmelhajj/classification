import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from RF_classifier.example import get_features

# get feature example
features, _, var_names = get_features()

# define inputs and output and split
df_x = features[var_names]
df_y = features['ref_class']
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.5, random_state=0)

# define empty array for rmse
rmse_array = []
estimators_nb = range(1, 150, 1)

for est in estimators_nb:
    # train
    model = RandomForestClassifier(n_estimators=est, random_state=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # get score
    fscore = f1_score(y_test, y_pred, average='macro') * 100
    print(fscore)

    # append stats
    rmse_array.append(fscore)

if __name__ == '__main__':
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10), sharex='all')
    ax.plot(estimators_nb, rmse_array, '-o')

    ax.set_ylabel('F_score', fontsize=20)
    ax.set_xlabel('Estimators NB', fontsize=20)

    # fmt axes
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    plt.show()
