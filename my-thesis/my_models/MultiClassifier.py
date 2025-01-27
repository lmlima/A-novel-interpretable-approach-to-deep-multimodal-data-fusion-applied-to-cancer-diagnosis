"""
Author: Leandro Lima
"""

from torch import classes
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np

from sklearn.metrics import balanced_accuracy_score, classification_report
import sys
from constants import RAUG_PATH

sys.path.insert(0, RAUG_PATH)
from raug.metrics import Metrics
from raug.checkpoints import load_model



class MultiClassifier():
    def __init__(self, backbone_model, classifier_name="XGBoost", checkpoint_path=None, class_names=None, device="cpu", features="joint",
                 hyp_params=None):
        assert class_names is not None, "class_names must be provided"
        n_classes = len(class_names)

        self.classifier_name = classifier_name
        if self.classifier_name == "XGBoost":
            self.model = XGBClassifier(objective="multi:softmax", num_class=n_classes, eval_metric="mlogloss")
        elif self.classifier_name == "LightGBM":
            self.model = LGBMClassifier(objective="multi:softmax", num_class=n_classes, eval_metric="mlogloss")

        self.backbone_model = backbone_model
        if checkpoint_path is not None:
            self.backbone_model = load_model(checkpoint_path, self.backbone_model)
        self.device = device

        self.early_stop = 15
        self.features = features

    def fit(self, train_data_loader, val_data_loader):

        train_concat_features, y_train, _ = self.prepare_data(train_data_loader)
        val_concat_features, y_val, _ = self.prepare_data(val_data_loader)

        # TODO: Train the classifier.
        self.model.fit(
            train_concat_features, y_train,
            eval_set=[(val_concat_features, y_val)],
            early_stopping_rounds=self.early_stop
        )

    def prepare_data(self, data_loader):
        # TODO: Load image and extract features on backbone model.

        for batch_idx, (img, target, meta_data, sample_name) in enumerate(data_loader):
            img = img.to(self.device)
            # meta_data = meta_data.to(self.device)
            if self.features == "late":
                feat_out = self.backbone_model.backbone_features(img).detach().cpu().numpy()
                feat_out = feat_out.squeeze((-1, -2))
                try:
                    concat_features = np.concatenate((feat_out, meta_data), axis=1)
                    concat_samples = np.concatenate((concat_samples, concat_features), axis=0)
                    concat_targets = np.concatenate((concat_targets, target), axis=0)
                    concat_name = np.concatenate((concat_name, sample_name), axis=0)
                except:
                    concat_features = np.concatenate((feat_out, meta_data), axis=1)
                    concat_samples = concat_features
                    concat_targets = target
                    concat_name = sample_name
            elif self.features == "mixed":
                feat_out = self.backbone_model.backbone_features(img)
                feat_out = self.backbone_model.reducer_block(feat_out.squeeze(-1).squeeze(-1))
                feat_out = self.backbone_model.classifier(feat_out).detach().cpu().numpy()
                # feat_out (batch_size, num_classes)
                try:
                    concat_features = np.concatenate((feat_out, meta_data), axis=1)
                    concat_samples = np.concatenate((concat_samples, concat_features), axis=0)
                    concat_targets = np.concatenate((concat_targets, target), axis=0)
                    concat_name = np.concatenate((concat_name, sample_name), axis=0)
                except:
                    concat_features = np.concatenate((feat_out, meta_data), axis=1)
                    concat_samples = concat_features
                    concat_targets = target
                    concat_name = sample_name
            elif self.features == "joint":
                meta_data = meta_data.to(self.device)
                feat_out = self.backbone_model.backbone_fusion(img, meta_data).detach().cpu().numpy()
                feat_out = feat_out.squeeze((-1, -2))
                try:
                    concat_samples = np.concatenate((concat_samples, feat_out), axis=0)
                    concat_targets = np.concatenate((concat_targets, target), axis=0)
                    concat_name = np.concatenate((concat_name, sample_name), axis=0)
                except:
                    concat_samples = feat_out
                    concat_targets = target
                    concat_name = sample_name
                
        return concat_samples, concat_targets, concat_name

    def eval(self, test_data_loader, save_pred=False,
             partition_name='Test', metrics_to_comp=('accuracy'), class_names=None, metrics_options=None,
             apply_softmax=True, verbose=True, full_path_pred=None):
        test_concat_features, y_true, test_samples_name = self.prepare_data(test_data_loader)

        y_pred = self.model.predict_proba(test_concat_features)

        # Setting the metrics object
        metrics = Metrics(metrics_to_comp, class_names, metrics_options)

        metrics.update_scores(y_true, y_pred, test_samples_name)

        # Getting the metrics
        metrics.compute_metrics()

        if save_pred or metrics.metrics_names is None:
            if full_path_pred is None:
                metrics.save_scores()
            else:
                _spt = full_path_pred.split('/')
                _folder = "/".join(_spt[0:-1])
                _p = _spt[-1]
                metrics.save_scores(folder_path=_folder, pred_name=_p)

        if verbose:
            print('- {} metrics:'.format(partition_name))
            metrics.print()

        return metrics.metrics_values
        # balanced_accuracy_score(y_true, y_pred)
        # classification_report(y_true, y_pred, target_names=target_names)

