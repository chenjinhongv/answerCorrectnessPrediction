import os
import pickle
import pandas as pd
from gensim.models import Word2Vec


CACHE = "F:/machineLearnCamp/Project/answerCorrectnessPrediction/cache/"
ORIDATAPATH = "F:/machineLearnCamp/Project/answerCorrectnessPrediction/oriData/"


def get_questions(ori_data_path=ORIDATAPATH):
    questions = pd.read_csv(ori_data_path + "questions.csv", dtype={"question_id": 'int16', "correct_answer": "int8"})
    return questions


def get_lectures(ori_data_path=ORIDATAPATH):
    lectures = pd.read_csv(ori_data_path + "lectures.csv", dtype={"lecture_id": "int16"})
    return lectures


def get_train_df(ori_data_path=ORIDATAPATH):
    with open(ori_data_path + "train.pkl", "rb") as f:
        train_df = pickle.load(f)
    return train_df


def get_sentences(df, group_columns, target_columns):
    sentences = []
    for g in df.groupby([group_columns]):
        sentence = [str(i) for i in list(g[1][target_columns])]
        sentences.append(sentence)
    return sentences


def get_w2v_feat(sentences, index_name, vec_size, prefix):
    model = Word2Vec(sentences=sentences, size=vec_size, window=5, min_count=1, workers=16)

    # vec字典表示
    vec_dict = {int(x): model.wv[x] for x in model.wv.vocab.keys()}
    # 创建为dataframe
    df = pd.DataFrame(vec_dict)
    res = pd.DataFrame(df.values.T, columns=df.index, index=df.columns)
    res.columns = [prefix + "_" + str(i) for i in range(vec_size)]
    res.reset_index(inplace=True)
    res.rename(columns={"index": index_name}, inplace=True)
    del df
    del vec_dict

    return res


def get_question_tags(cache=CACHE, regain=False):
    """
    :param cache:缓存路径
    :param regain:是否重新生成
    :return: 返回question（相同question存在多行） 和 tag 1v1 数据框
    """
    if os.path.exists(cache+"question_tags.csv") and not regain:
        res = pd.read_csv(cache+"question_tags.csv")
    else:
        questions = get_questions()["tags"]
        question_id = []
        tag_list = []
        for q_id in list(questions.index):
            tags = str(questions[q_id])
            for tag in tags.split(" "):
                question_id.append(q_id)
                tag_list.append(tag)

        res = pd.DataFrame({"content_id": question_id, "tag": tag_list})

        res.to_csv(cache+"question_tags.csv", index=False)

    return res


def get_sample(cache=CACHE, ori_data_path=ORIDATAPATH, tail=0, span=10):
    if os.path.exists(cache+"sample_{}_{}.pkl".format(tail, span)):
        with open(cache+"sample_{}_{}.pkl".format(tail, span), "rb") as f:
            sample = pickle.load(f)
        with open(cache+"behave_for_sample_{}_{}.pkl".format(tail, span), "rb") as f:
            behave_for_sample = pickle.load(f)
    else:
        # 读取数据
        train_df = get_train_df(ori_data_path=ori_data_path)

        # 序列化数据读取问题，临时处理方案
        train_df["prior_question_had_explanation"].fillna(False, inplace=True)
        train_df.astype({"prior_question_had_explanation": "int8"})
        # # 采样用户sample_rate
        # # 用户列表
        # if sample_rate < 1:
        #     user = train_df[["user_id"]].drop_duplicates(subset=["user_id"])
        #
        #     # 随机采样
        #     from sklearn.model_selection import train_test_split
        #     user_left, user_subset, c, d = train_test_split(user, user, test_size=sample_rate, shuffle=True)
        #
        #     # 过滤用户行为数据
        #     train_df = pd.merge(user_subset, train_df, on=["user_id"], how="left")

        # sample & behave_for_sample
        sample = train_df[train_df.content_type_id == 0].groupby(["user_id"]).tail(span+tail).copy()
        behave_for_sample = train_df[~train_df["row_id"].isin(sample["row_id"])].copy()
        sample = sample.groupby(["user_id"]).head(span).copy()
        del train_df

        # save the deal
        with open(cache+"sample_{}_{}.pkl".format(tail, span), "wb") as f:
            pickle.dump(sample, f)
        with open(cache+"behave_for_sample_{}_{}.pkl".format(tail, span), "wb") as f:
            pickle.dump(behave_for_sample, f)

    # print deal info
    print("get len of sample_{}_{}:{}".format(tail, span, len(sample)))
    print("get len of behave_for_sample_{}_{}:{}".format(tail, span, len(behave_for_sample)))

    return sample, behave_for_sample


if __name__ == "__main__":
    # 创建5份样本
    # for i in range(5):
    #     get_sample(tail=i*10)

    get_question_tags(regain=True)
