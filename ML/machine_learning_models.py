# imports
import os
import numpy as np
import pandas as pd
import itertools as it
from scipy import stats
from scipy.sparse import csr_matrix
# Sklearn
from sklearn import neighbors, metrics
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from regression_training_size.sveta.sveta.svm import ExplainingSVR

from ML.ml_utils import tanimoto_from_dense

import warnings

os.environ["TF_DETERMINISTIC_OPS"] = "1"
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

class MLModel:
    def __init__(self, training_data, ml_algorithm, opt_metric_name='MAE',
                 reg_class="regression", parameters='grid', random_seed=2002, data_type='balanced'):

        self.training_data = training_data
        self.ml_algorithm = ml_algorithm
        self.reg_class = reg_class
        self.seed = random_seed
        self.parameters = parameters
        self.opt_metric = opt_metric_name
        self.data_type = data_type
        self.h_parameters = self.hyperparameters()
        self.base_model = self.base_model()
        self.opt_results = self.hp_grid_search()
        self.best_params = self.optimal_parameters()
        self.model = self.final_model()

    def hyperparameters(self):
        if self.parameters == "grid":

            if self.reg_class == "regression":
                if self.ml_algorithm == "MR":
                    return {'strategy': ['median']
                            }
                elif self.ml_algorithm == "SVR":
                    return {'C': [1, 10, 100, 10000],
                            'kernel': [tanimoto_from_dense],
                            }
                elif self.ml_algorithm == "ExplainingSVR":
                    return {'C': [1, 10, 100, 10000],
                        }
                elif self.ml_algorithm == "RFR":
                    return {'n_estimators': [50, 100, 200],
                            'max_features': ['sqrt', 'log2'],
                            'min_samples_split': [2, 3, 5, 10],
                            'min_samples_leaf': [1, 2, 5, 10],

                            }
                elif self.ml_algorithm == "kNN":
                    return {"n_neighbors": [3],
                            'metric': ['jaccard'],
                            }
                elif self.ml_algorithm == "1-NN":
                    return {"n_neighbors": [1],
                            'metric': ['jaccard'],
                            }

            if self.reg_class == "classification":
                if self.ml_algorithm == "SVM":
                    return {'C': [1, 10, 100, 10000],
                            'kernel': [tanimoto_from_dense],
                            }
                elif self.ml_algorithm == "RFC":
                    return {'n_estimators': [25, 50, 100, 200, 400],
                            'max_features': ['sqrt', 'log2'],
                            'min_samples_split': [2, 3, 5, 10],
                            'min_samples_leaf': [1, 2, 5, 10],
                            }
                elif self.ml_algorithm == "kNN":
                    return {"n_neighbors": [3, 5],
                            'metric': ['jaccard'],
                            }
                elif self.ml_algorithm == "1-NN":
                    return {"n_neighbors": [1],
                            'metric': ['jaccard'],
                            }

    def base_model(self):

        global model
        if self.reg_class == "regression":
            if self.ml_algorithm == "MR":
                model = DummyRegressor()
            elif self.ml_algorithm == "SVR":
                model = SVR()
            elif self.ml_algorithm == "ExplainingSVR":
                model = ExplainingSVR()
            elif self.ml_algorithm == "RFR":
                model = RandomForestRegressor(random_state=self.seed)
            elif self.ml_algorithm == "kNN":
                model = neighbors.KNeighborsRegressor(metric='jaccard')
            elif self.ml_algorithm == "1-NN":
                model = neighbors.KNeighborsRegressor(metric='jaccard')

        return model

    def hp_grid_search(self):

        # initialize hyper-parameters
        hyperparameters = self.h_parameters

        # Build grid for all parameters
        parameter_grid = {n: {name: value for name, value in zip(hyperparameters.keys(), comb)}
                          for n, comb in enumerate(it.product(*list(hyperparameters.values())), 1)}

        df_evaluation = pd.DataFrame()

        opt_metric_name = self.opt_metric

        for i, idx_params in enumerate(parameter_grid):

            cur_params = parameter_grid[idx_params]

            results_cv = []
            if self.data_type == 'balanced':
                data_splitter = StratifiedShuffleSplit(n_splits=3, random_state=42, test_size=0.5)
            else:
                data_splitter = ShuffleSplit(n_splits=3, random_state=42, test_size=0.5)

            # select training and validation data
            for trial, (train_idx, valid_idx) in enumerate(data_splitter.split(X=self.training_data.features,
                                                                               y=self.training_data.potency_classes)):
                train_data = self.training_data[train_idx]

                valid_data = self.training_data[valid_idx]

                model_hp = self.base_model.set_params(**cur_params)

                if self.ml_algorithm == "ExplainingSVR":
                    model_hp.fit(csr_matrix(train_data.features), train_data.labels)
                    if opt_metric_name == 'MAE':
                        results_cv_val = {'mae': mean_absolute_error(valid_data.labels,
                                                                 model_hp.predict(csr_matrix(valid_data.features)))}
                else:
                    model_hp.fit(train_data.features, train_data.labels)
                    if opt_metric_name == 'MAE':
                        results_cv_val = {'mae': mean_absolute_error(valid_data.labels,
                                                                 model_hp.predict(valid_data.features))}
                results_cv.append(results_cv_val)

                del model_hp

            valid_avg_score = sum(item.get('mae', 0) for item in results_cv) / len(results_cv)

            # print(val_score)
            df_evaluation = df_evaluation.append({
                'params': cur_params,
                opt_metric_name: valid_avg_score,
                'model': self.ml_algorithm},
                ignore_index=True)

        return df_evaluation

    def optimal_parameters(self):

        df_eval = self.opt_results
        min_valid_score = df_eval[self.opt_metric].min()
        df_min_score = df_eval[df_eval[self.opt_metric] == min_valid_score]
        best_params = df_min_score.iloc[0]['params']
        return best_params

    def final_model(self):
        model_ = self.base_model.set_params(**self.best_params)
        if self.ml_algorithm == "ExplainingSVR":
            return model_.fit(csr_matrix(self.training_data.features), self.training_data.labels)
        else:
            return model_.fit(self.training_data.features, self.training_data.labels)

class Model_Evaluation:
    def __init__(self, model, data, tr_data, model_id=None, reg_class="regression", ):
        self.reg_class = reg_class
        self.model_id = model_id
        self.model = model
        self.data = data
        self.tr_data = tr_data
        self.labels, self.y_pred, self.predictions = self.model_predict(data)
        self.pred_performance = self.prediction_performance(data)

    def model_predict(self, data):

        if self.reg_class == "regression":
            if self.model_id == "ExplainingSVR":
                y_prediction = self.model.model.predict(csr_matrix(data.features))
            else:
                y_prediction = self.model.model.predict(data.features)

            labels = self.data.labels

            predictions = pd.DataFrame(list(zip(data.cid, labels, y_prediction, data.smiles)),
                                       columns=["cid", "Experimental", "Predicted", 'smiles'])
            predictions['Target ID'] = data.target[0]
            predictions['Algorithm'] = self.model.ml_algorithm
            predictions["Target ID"] = predictions["Target ID"].map(lambda x: x.lstrip("CHEMBL").rstrip(""))
            predictions['Residuals'] = [label_i - prediction_i for label_i, prediction_i in zip(labels, y_prediction)]

            if self.model.ml_algorithm == '1-NN':
                nn = self.model.model.kneighbors(X=data.features, n_neighbors=1, return_distance=False)
                predictions['1-NN'] = self.tr_data[nn].cid
                predictions['1-NN_smiles'] = self.tr_data[nn].smiles
            else:
                predictions['1-NN'] = np.nan
                predictions['1-NN_smiles'] = np.nan

            return labels, y_prediction, predictions

        elif self.reg_class == 'classification':

            y_prediction = self.model.model.predict(data.features)
            labels = self.data.labels

            predictions = pd.DataFrame(list(zip(data.cid, labels, y_prediction)),
                                       columns=["cid", "Experimental", "Predicted"])
            predictions['Target ID'] = data.target[0]
            predictions['Algorithm'] = self.model.ml_algorithm
            predictions["Target ID"] = predictions["Target ID"].map(lambda x: x.lstrip("CHEMBL").rstrip(""))

            return labels, y_prediction, predictions

    def prediction_performance(self, data, nantozero=False) -> pd.DataFrame:

        if self.reg_class == "regression":

            labels = self.labels
            pred = self.y_pred

            fill = 0 if nantozero else np.nan
            if len(pred) == 0:
                mae = fill
                mse = fill
                rmse = fill
                r2 = fill
                r = fill
            else:
                mae = mean_absolute_error(labels, pred)
                mse = metrics.mean_squared_error(labels, pred)
                rmse = metrics.mean_squared_error(labels, pred, squared=False)
                r2 = metrics.r2_score(labels, pred)
                r = stats.pearsonr(labels, pred)[0]

            target = data.target[0]
            model_name = self.model.ml_algorithm

            result_list = [{"MAE": mae,
                            "MSE": mse,
                            "RMSE": rmse,
                            "R2": r2,
                            "r": r,
                            "Dataset size": len(labels),
                            "Target ID": target,
                            "Algorithm": model_name}
                           ]

            # Prepare result dataset
            results = pd.DataFrame(result_list)
            results["Target ID"] = results["Target ID"].map(lambda x: x.lstrip("CHEMBL").rstrip(""))
            results.set_index(["Target ID", "Algorithm", "Dataset size"], inplace=True)
            results.columns = pd.MultiIndex.from_product([["Value"], ["MAE", "MSE", "RMSE", "R2", "r"]],
                                                         names=["Value", "Metric"])
            results = results.stack().reset_index().set_index("Target ID")

            return results

        elif self.reg_class == 'classification':

            labels = self.labels
            pred = self.y_pred

            target = data.target[0]
            model_name = self.model.ml_algorithm

            result_list = [{"MCC": metrics.matthews_corrcoef(labels, pred),
                            "F1": metrics.f1_score(labels, pred),
                            "AUC": metrics.roc_auc_score(labels, pred),
                            "BA": metrics.balanced_accuracy_score(labels, pred),
                            "Accuracy": metrics.accuracy_score(labels, pred),
                            "Precision": metrics.precision_score(labels, pred),
                            "Recall": metrics.recall_score(labels, pred),
                            "Average Precision": metrics.average_precision_score(labels, pred),
                            "Dataset size": len(labels),
                            "Target ID": target,
                            "Algorithm": model_name,
                            "pos_true": len([x for x in labels if x == 1]),
                            "neg_true": len([x for x in labels if x == 0]),
                            "pos_predicted": len([x for x in pred if x == 1]),
                            "neg_predicted": len([x for x in pred if x == 0])
                            }]

            # Prepare result dataset
            results = pd.DataFrame(result_list)
            results = results[
                "Target ID", "Algorithm", "Dataset size", "MCC", "F1", "AUC", "BA", "Accuracy", "Precision", "Recall", "Average Precision", "pos_true", "neg_true", "pos_predicted", "neg_predicted"]
            results["Target ID"] = results["Target ID"].map(lambda x: x.lstrip("CHEMBL").rstrip(""))
            results.set_index(["Target ID", "Algorithm", "Dataset size"], inplace=True)
            results.columns = pd.MultiIndex.from_product([["Value"], ["MCC", "F1", "AUC", "BA", "Accuracy",
                                                                      "Precision", "Recall", "Average Precision",
                                                                      "pos_true", "neg_true", "pos_predicted",
                                                                      "neg_predicted"]],
                                                         names=["Value", "Metric"])
            results = results.stack().reset_index().set_index("Target ID")

            return results


