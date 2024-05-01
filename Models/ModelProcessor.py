from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
import numpy as np
from numpy import interp
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, make_scorer



class ModelProcessor:
    def __init__(self,x_test,x_train,y_test,y_train,model):
        self.x_test = x_test
        self.x_train = x_train
        self.y_test = y_test
        self.y_train = y_train
        self.pipeline = None
        self.y_pred = None
        self.model = model

    def CreatePipeline(self, *pipeline_steps):
        self.pipeline_steps = pipeline_steps

    def PrintBestModelParams(self,scorer,param_grid):

        gs = GridSearchCV(estimator=self.model,
                          param_grid=param_grid,
                          scoring=scorer,
                          cv=3,
                          n_jobs=-1)

        gs = gs.fit(self.x_train, self.y_train)
        print(f"{type(self.model).__name__} f1 score: ", gs.best_score_)

        print(f"{type(self.model).__name__} best params: ", gs.best_params_)

        print(type(gs.best_params_))

    def FitAndPredict(self, pipeline):

        pipeline.fit(X=self.x_train,y=self.y_train)

        self.y_pred = pipeline.predict(self.x_test)

        confmat = confusion_matrix(y_true=self.y_test, y_pred=self.y_pred)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
        ax.xaxis.set_ticks_position('bottom')

        plt.xlabel('Predicted label')
        plt.ylabel('True label')

        plt.tight_layout()
        plt.show()

    def print_scores(self):
        acc_val = accuracy_score(y_true=self.y_test, y_pred=self.y_pred)
        print(f'Accuracy: {acc_val:.3f}')

        pre_val = precision_score(y_true=self.y_test, y_pred=self.y_pred, average='weighted')
        print(f'Precision: {pre_val:.3f}')

        rec_val = recall_score(y_true=self.y_test, y_pred=self.y_pred, average='weighted')
        print(f'Recall: {rec_val:.3f}')

        f1_val = f1_score(y_true=self.y_test, y_pred=self.y_pred, average='weighted')
        print(f'F1: {f1_val:.3f}')

    def k_folds(self,pipeline):

        X_train_vals = self.x_train.values
        y_train_vals = self.y_train.values
        X_train2 = X_train_vals[:, [4, 14]]

        cv = list(StratifiedKFold(n_splits=3).split(self.x_train, self.y_train))

        fig = plt.figure(figsize=(7, 5))

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []

        for i, (train, test) in enumerate(cv):
            probas = pipeline.fit(X_train2[train],
                                 y_train_vals[train]).predict_proba(X_train2[test])

            fpr, tpr, thresholds = roc_curve(y_train_vals[test],
                                             probas[:, 1],
                                             pos_label=1)

            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr,
                     tpr,
                     label=f'ROC fold {i + 1} (area = {roc_auc:.2f})')

        plt.plot([0, 1],
                 [0, 1],
                 linestyle='--',
                 color=(0.6, 0.6, 0.6),
                 label='Random guessing (area = 0.5)')

        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--',
                 label=f'Mean ROC (area = {mean_auc:.2f})', lw=2)
        plt.plot([0, 0, 1],
                 [0, 1, 1],
                 linestyle=':',
                 color='black',
                 label='Perfect performance (area = 1.0)')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc='lower right')
        plt.title("Logistic Regression")

        plt.tight_layout()
        plt.show()