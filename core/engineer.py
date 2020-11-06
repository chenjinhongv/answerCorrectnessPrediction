import lightgbm as lgb
from lightgbm import Dataset
import pandas as pd

PARAMS = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.2,
    "max_depth": 6,
    "num_leaves": 50,
    #     "feature_fraction":0.8,
    #     "bagging_fraction":0.8,
    #     "bagging_freq":15,
    "device": "gpu"
}


# a features selector for lightgbm
def lgbm_feat_selector(train_x, train_y, valid_x, valid_y, params=PARAMS, drop_size=1, keep_size=20):
    """
    :param train_x:
    :param train_y:
    :param valid_x:
    :param valid_y:
    :param params:
    :param drop_size:
    :param keep_size:
    :return:list[dict({"round":select_round,
                    "features":left_features_this_round,
                    "train_round":best_iteration_of_model,
                    "auc":auc-score})]
    """
    res = []
    feat_list = list(train_x.columns)
    round_ = 0
    while len(feat_list) > keep_size:
        print("-"*25 + "selector round {}".format(round_) + "-"*25)
        sub_col_train_x = train_x[feat_list]
        sub_col_valid_x = valid_x[feat_list]
        data_train = Dataset(sub_col_train_x.values, train_y.values)
        data_valid = Dataset(sub_col_valid_x.values, valid_y.values)

        print("-"*25 + "training" + "-"*25)
        cv_res = lgb.cv(params=params, train_set=data_train, nfold=4, stratified=True, shuffle=True,
                        early_stopping_rounds=10, num_boost_round=1000)

        train_score = max(cv_res["auc-mean"])
        iteration = len(cv_res["auc-mean"])
        print("-"*25 + "saving result" + "-"*25)
        res.append({"round": round_, "features": feat_list, "train_round": iteration, "auc": train_score})
        # drop tail features
        model = lgb.train(params=params, train_set=data_train, num_boost_round=iteration,
                          valid_sets=[data_train, data_valid])
        del data_train, data_valid
        print("-"*25 + "dropping tail {} features".format(drop_size) + "-"*25)
        feature_importance = pd.Series(model.feature_importance(), index=sub_col_train_x.columns).sort_values()
        tail_features = list(feature_importance.head(drop_size).index)
        feat_list = list(set(feat_list) - set(tail_features))

        round_ += 1

    return res
