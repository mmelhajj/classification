import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split

from RF_classifier.example import get_example
from info import outputs
from viualization.common import plot_conf_matrix

# get feature example
features, _, var_names = get_example()

# define inputs and output and split
df_x = features[var_names]
df_y = features[['ref_class1', 'ref_class2']]
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3, random_state=0)

# fit RF regressor classifier: https://towardsdatascience.com/classification-with-random-forests-in-python-29b8381680ed
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# save model
joblib.dump(model, "./random_forest.joblib")

# load the model and predict
model = joblib.load("./random_forest.joblib")
y_pred = model.predict(X_test)

if __name__ == '__main__':
    # get score and plot the confusion matrix
    for i, (true, pred) in enumerate(zip([y_test['ref_class1'], y_test['ref_class2']], [y_pred[:, 0], y_pred[:, 1]])):
        fig_, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15))
        # get label
        labels = true.unique().tolist()
        labels.sort()

        print(i, labels)

        # get score
        fscore = '%.2f' % (f1_score(true, pred, average='macro') * 100)
        kappa = '%.2f' % (cohen_kappa_score(true, pred) * 100)

        # conf matrix
        cm = confusion_matrix(true, pred)

        # add title
        ax.set_title(f'season{i + 1}/ fscore:{fscore} % - Kappa: {kappa} %', fontsize=20)

        # plot
        plot_conf_matrix(cm, 'predicted', 'actual', labels, 20, ax, font_scale=2)
        # save figure
        plt.savefig(f"{outputs}/confusion_season{i + 1}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

    # get and plot importance
    importance = model.feature_importances_
    importance = pd.DataFrame(importance, columns=['importance'], index=X_train.columns)
    importance = importance.sort_values(['importance'])

    fig_, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10), sharex='all')
    ax.barh(importance.index, importance['importance'])

    # # fmt axes
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=10, rotation=0)
    ax.set_xlabel('Performance', fontsize=20)

    # save figure
    plt.savefig(f"{outputs}/importance.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
