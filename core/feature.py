from core.base import *
import numpy as np


def get_usr_acc_feat(cache=CACHE, tail=0, span=10, regain=False):
    """
    :param cache:缓存路径
    :param tail:排除每个用户末尾tail个回答记录
    :param span:排除每个用户末尾span个作为样本的回答记录
    :param regain:是否重新生成
    :return:用户累积行为特征
    """
    if os.path.exists(cache + "usr_acc_feat_for_sample_{}_{}.csv".format(tail, span)) and not regain:
        res = pd.read_csv(cache + "usr_acc_feat_for_sample_{}_{}.csv".format(tail, span))
    else:
        # load data
        if span == 0:
            df = get_train_df()
        else:
            df = get_sample(tail=tail, span=span)[1]
        questions = get_questions()
        lectures = get_lectures()

        res = df[["user_id"]].drop_duplicates(subset=["user_id"])
        df["prior_question_had_explanation"].fillna(False, inplace=True)
        df.astype({"prior_question_had_explanation": "int8"})

        # 用户做题数量，准确率
        tmp = df.loc[df.content_type_id == 0, :].groupby(["user_id"]).agg(
            {"answered_correctly": ["count", "mean"]})
        tmp.columns = ["usr_ans_count", "usr_accuracy"]
        res = pd.merge(res, tmp, on=["user_id"], how="left")

        # 用户查看解释的比率
        tmp = df.loc[df.content_type_id == 0, :].groupby(["user_id"]).agg({"prior_question_had_explanation": ["mean", "sum"]})
        tmp.columns = ["usr_had_explanation_rate", "usr_had_explanation_count"]
        res = pd.merge(res, tmp, on=["user_id"], how="left")

        # 用户对各种part（top_level）的题目做题数量，准确率
        user_question_act = df.loc[df.content_type_id == 0, :]
        user_question_act = pd.merge(user_question_act, questions.rename(columns={"question_id": "content_id"}),
                                     on=["content_id"], how="left")
        tmp = pd.pivot_table(user_question_act, index=["user_id"], columns=["part"], values=["answered_correctly"],
                             aggfunc=["count", "mean"])
        tmp.columns = ["usr_part" + str(x) + "ans_count" for x in range(1, 8)] + \
                      ["usr_part" + str(x) + "ans_accuracy" for x in range(1, 8)]
        res = pd.merge(res, tmp, on=["user_id"], how="left")
        del user_question_act

        # 用户听课数量
        tmp = df.loc[df.content_type_id == 1, :].groupby(["user_id"]).agg({"answered_correctly": "count"})
        tmp.columns = ["usr_lecture_count"]
        res = pd.merge(res, tmp, on=["user_id"], how="left")
        res["usr_lecture_count"].fillna(0, inplace=True)

        # 用户听各种意图的课的次数
        user_lecture_act = df.loc[df.content_type_id == 1, :]
        user_lecture_act = pd.merge(user_lecture_act, lectures.rename(columns={"lecture_id": "content_id"}),
                                    on=["content_id"], how="left")
        user_lecture_act = pd.concat([user_lecture_act, pd.get_dummies(user_lecture_act.type_of)], axis=1)
        tmp = user_lecture_act.groupby(["user_id"]).agg({"concept": "sum",
                                                         "intention": "sum",
                                                         "solving question": "sum"})
        tmp.columns = ["usr_" + x + "lecture_count" for x in ["concept", "intention", "solvingquestion"]]
        res = pd.merge(res, tmp, on=["user_id"], how="left")

        # 用户听各种part（top_level）的课的次数
        user_lecture_act = pd.concat([user_lecture_act, pd.get_dummies(user_lecture_act.part, prefix="part")], axis=1)
        tmp = user_lecture_act.groupby(["user_id"]).agg({"part_1": "sum",
                                                         "part_2": "sum",
                                                         "part_3": "sum",
                                                         "part_4": "sum",
                                                         "part_5": "sum",
                                                         "part_6": "sum",
                                                         "part_7": "sum"})
        tmp.columns = ["usr_" + x + "lecture_count" for x in
                       ["part_1", "part_2", "part_3", "part_4", "part_5", "part_6", "part_7"]]
        res = pd.merge(res, tmp, on=["user_id"], how="left")
        del user_lecture_act

        # 用户最后10次，20次，50次回答的准确率
        tmp = df.loc[df.content_type_id == 0, :].groupby(["user_id"]).tail(10).groupby(["user_id"]).agg(
            {"answered_correctly": lambda x: sum(x) / len(x) if len(x) == 10 else np.NaN})
        tmp.columns = ["usr_last_10_ans_accuracy"]
        res = pd.merge(res, tmp, on=["user_id"], how="left")
        tmp = df.loc[df.content_type_id == 0, :].groupby(["user_id"]).tail(20).groupby(["user_id"]).agg(
            {"answered_correctly": lambda x: sum(x) / len(x) if len(x) == 20 else np.NaN})
        tmp.columns = ["usr_last_20_ans_accuracy"]
        res = pd.merge(res, tmp, on=["user_id"], how="left")
        tmp = df.loc[df.content_type_id == 0, :].groupby(["user_id"]).tail(50).groupby(["user_id"]).agg(
            {"answered_correctly": lambda x: sum(x) / len(x) if len(x) == 50 else np.NaN})
        tmp.columns = ["usr_last_50_ans_accuracy"]
        res = pd.merge(res, tmp, on=["user_id"], how="left")

        res.to_csv(cache + "usr_acc_feat_for_sample_{}_{}.csv".format(tail, span), index=False)

    # print deal info
    print("succeed get usr_acc_feat_for_sample_{}_{}".format(tail, span))

    return res


def get_usr_act_time_feat(cache=CACHE, tail=0, span=10, regain=False):
    """
    :param cache:缓存路径
    :param tail:排除每个用户末尾tail个回答记录
    :param span:排除每个用户末尾span个作为样本的回答记录,span=0时,产出用于提交的全集特征
    :param regain:是否重新生成
    :return:用户行为的时间特征
    """
    if os.path.exists(cache + "usr_act_time_feat_for_sample_{}_{}.csv".format(tail, span)) and not regain:
        res = pd.read_csv(cache + "usr_act_time_feat_for_sample_{}_{}.csv".format(tail, span))
    else:
        # load data
        if span == 0:
            df = get_train_df()
        else:
            df = get_sample(tail=tail, span=span)[1]

        # get usr
        res = df[["user_id"]].drop_duplicates(subset=["user_id"])

        # # 用户最近一次回答问题的时间
        # tmp = df.loc[df.content_type_id == 0, :].groupby(["user_id"]).agg({"timestamp": "max"})
        # tmp.columns = ["user_last_answer_time"]
        # res = pd.merge(res, tmp, on=["user_id"], how="left")

        # # 用户最近一次听课的时间
        # tmp = df.loc[df.content_type_id == 1, :].groupby(["user_id"]).agg({"timestamp": "max"})
        # tmp.columns = ["user_last_lecture_time"]
        # res = pd.merge(res, tmp, on=["user_id"], how="left")

        # 用户行为间隔的最大，最小，均值，标准差（用于衡量用户学习频率稳定程度）
        df["act_gap"] = df.groupby(["user_id"])["timestamp"].diff()
        tmp = df.groupby(["user_id"]).agg({"act_gap": ["max", "min", "mean", np.std]})
        tmp.columns = ["usr_act_gap_max", "usr_act_gap_min", "usr_act_gap_mean", "usr_act_gap_std"]
        res = pd.merge(res, tmp, on=["user_id"], how="left")

        # # 用户倒数第5次，第10次，第20次，第50次，第100次回答的时间间隔最大，最小，均值，标准差
        for tail_i in [5, 10, 20, 50]:
            tmp = df.loc[df.content_type_id == 0, :].groupby(["user_id"]).head(tail_i)
            tmp = tmp.groupby(["user_id"]).agg({"act_gap": [lambda x: min(x) if len(x) >= tail_i else np.NaN,
                                                            lambda x: max(x) if len(x) >= tail_i else np.NaN,
                                                            lambda x: np.mean(x) if len(x) >= tail_i else np.NaN,
                                                            lambda x: np.std(x) if len(x) >= tail_i else np.NaN]})
            tmp.columns = ["usr_last_" + str(tail_i) + "_ans_gap_" + x for x in ["min", "max", "mean", "std"]]
            res = pd.merge(res, tmp, on=["user_id"], how="left")

        res.to_csv(cache + "usr_act_time_feat_for_sample_{}_{}.csv".format(tail, span), index=False)

    # print deal info
    print("succeed get usr_act_time_feat_for_sample_{}_{}".format(tail, span))

    return res


def get_question_acc_feat(cache=CACHE, regain=False):
    """
    :param cache:缓存路径
    :param regain:是否重新生成
    :return: 用户在问题上的行为累积特征
    """
    if os.path.exists(cache+"question_acc_feat.csv") and not regain:
        res = pd.read_csv(cache+"question_acc_feat.csv")
    else:
        # load data
        df = get_train_df()

        # 问题被回答的次数，问题回答的准确率（衡量问题热度&难度）
        res = df.loc[df.content_type_id == 0, :].groupby(["content_id"]).agg({"answered_correctly": ["count", "mean"]})
        res.columns = ["question_ans_count", "question_ans_accuracy"]
        res.reset_index(inplace=True)

        res.to_csv(cache+"question_acc_feat.csv", index=False)

    # print deal info
    print("succeed get question_acc_feat")

    return res


def get_usr_vec_ans(cache=CACHE, size=5, regain=False):
    """
    :param cache:缓存路径
    :param size:词向量长度：size*2
    :param regain:是否重新生成
    :return:用户打对（错）相同题目的特征的词向量表示
    """
    if os.path.exists(cache+"usr_vec_ans_{}.csv".format(size)) and not regain:
        res = pd.read_csv(cache+"usr_vec_ans_{}.csv".format(size))
    else:
        # load data
        df = get_train_df()
        df_pos = df.loc[(df.content_type_id == 0) & (df.answered_correctly == 1), ["user_id", "content_id"]]
        df_neg = df.loc[(df.content_type_id == 0) & (df.answered_correctly == 0), ["user_id", "content_id"]]

        # get pos ans usr vec
        sentences = get_sentences(df=df_pos, group_columns="content_id", target_columns="user_id")
        pos_ans_vec = get_w2v_feat(sentences=sentences, index_name="user_id", vec_size=size, prefix="usr_pos_ans_vec")

        # get neg ans usr vec
        sentences = get_sentences(df=df_neg, group_columns="content_id", target_columns="user_id")
        neg_ans_vec = get_w2v_feat(sentences=sentences, index_name="user_id", vec_size=size, prefix="usr_neg_ans_vec")

        res = pd.merge(pos_ans_vec, neg_ans_vec, on=["user_id"], how="outer")
        res.to_csv(cache+"usr_vec_ans_{}.csv".format(size), index=False)

    # print deal info
    print("succeed get usr_vec_ans_{}".format(size))

    return res


def get_question_vec_ans(cache=CACHE, size=5, regain=False):
    """
    :param cache:缓存路径
    :param size:词向量长度size*2
    :param regain:是否重新生成
    :return:被同一用户答对（错）的题目的特征的词向量表示
    """
    if os.path.exists(cache+"question_vec_ans_{}.csv".format(size)) and not regain:
        res = pd.read_csv(cache+"question_vec_ans_{}.csv".format(size))
    else:
        # load data
        df = get_train_df()
        df_pos = df.loc[(df.content_type_id == 0) & (df.answered_correctly == 1), ["user_id", "content_id"]]
        df_neg = df.loc[(df.content_type_id == 0) & (df.answered_correctly == 0), ["user_id", "content_id"]]

        # get pos ans question vec
        sentences = get_sentences(df=df_pos, group_columns="user_id", target_columns="content_id")
        pos_ans_vec = get_w2v_feat(sentences=sentences, index_name="content_id", vec_size=size,
                                   prefix="question_pos_ans_vec")

        # get neg ans question vec
        sentences = get_sentences(df=df_neg, group_columns="user_id", target_columns="content_id")
        neg_ans_vec = get_w2v_feat(sentences=sentences, index_name="content_id", vec_size=size,
                                   prefix="question_neg_ans_vec")

        # merge the result
        res = pd.merge(pos_ans_vec, neg_ans_vec, on=["content_id"], how="outer")

        # save
        res.to_csv(cache+"question_vec_ans_{}.csv".format(size), index=False)

    # print deal info
    print("succeed get question_vec_ans_{}".format(size))

    return res


def get_usr_vec_lecture(cache=CACHE, size=5, regain=False):
    """
    :param cache:缓存路径
    :param size:词向量长度size
    :param regain:是否重新生成
    :return:上过相同课程的用户特征size维词向量表示
    """
    if os.path.exists(cache+"usr_vec_lecture_{}.csv".format(size)) and not regain:
        res = pd.read_csv(cache+"usr_vec_lecture_{}.csv".format(size))
    else:
        # load data
        df = get_train_df()
        df = df.loc[df.content_type_id == 1, ["user_id", "content_id"]]

        # get vec
        sentences = get_sentences(df=df, group_columns="content_id", target_columns="user_id")
        res = get_w2v_feat(sentences=sentences, index_name="user_id", vec_size=5, prefix="usr_lecture_vec")

        res.to_csv(cache+"usr_vec_lecture_{}.csv".format(size), index=False)

    # print deal info
    print("succeed get usr_vec_lecture_{}".format(size))

    return res


def get_usr_qpart_acc_feat(cache=CACHE, tail=0, span=10, regain=False):
    """
    :param cache:缓存路径
    :param tail:排除每个用户末尾tail个回答记录
    :param span:排除每个用户末尾span个作为样本的回答记录,span=0时,产出用于提交的全集特征
    :param regain:是否重新生成
    :return: 用户对当前part的累积行为特征
    """
    if os.path.exists(cache+"usr_qpart_acc_feat_for_sample_{}_{}.csv".format(tail, span)) and not regain:
        res = pd.read_csv(cache+"usr_qpart_acc_feat_for_sample_{}_{}.csv".format(tail, span))
    else:
        # load data
        if span == 0:
            df = get_train_df()
        else:
            df = get_sample(tail=tail, span=span)[1]
        questions = get_questions()

        # 用户每个part的答题准确率，答题数量
        df = df.loc[df.content_type_id == 0, ["user_id", "content_id", "answered_correctly"]]
        df = pd.merge(df, questions[["question_id", "part"]].rename(columns={"question_id": "content_id"}),
                      on=["content_id"], how="left")

        res = df.groupby(["user_id", "part"]).agg({"answered_correctly": ["count","mean"]})
        res.columns = ["usr_curpart_count", "usr_curpart_accuracy"]

        res.reset_index(inplace=True)
        res.to_csv(cache+"usr_qpart_acc_feat_for_sample_{}_{}.csv".format(tail, span), index=False)

    # print deal info
    print("succeed get usr_qpart_acc_feat_for_sample_{}_{}".format(tail, span))

    return res


def get_usrxquestion_feat(cache=CACHE, tail=0, span=10, regain=False):
    """
    :param cache:缓存路径
    :param tail:排除每个用户末尾tail个回答记录
    :param span:排除每个用户末尾span个作为样本的回答记录,span=0时,产出用于提交的全集特征
    :param regain:是否重新生成
    :return: 用户与题目的交互记录
    """
    if os.path.exists(cache+"usrxquestion_feat_for_sample_{}_{}.csv".format(tail, span)) and not regain:
        res = pd.read_csv(cache+"usrxquestion_feat_for_sample_{}_{}.csv".format(tail, span))
    else:
        # load data
        if span == 0:
            df = get_train_df()
        else:
            df = get_sample(tail=tail, span=span)[1]

        # 提取用户题目交互记录
        res = df.loc[df.content_type_id == 0, ["user_id", "content_id"]].drop_duplicates(subset=["user_id", "content_id"])
        res["is_seen"] = 1
        res.to_csv(cache+"usrxquestion_feat_for_sample_{}_{}.csv".format(tail, span), index=False)

    # print deal info
    print("succeed get usrxquestion_feat_for_sample_{}_{}.csv".format(tail, span))

    return res


def get_task_id_tar_code(cache=CACHE, regain=False):
    """
    :param cache:结果保存路径
    :param regain:是否重新生成
    :return:返回task_container_id 的target_encode
    """
    if os.path.exists(cache+"task_id_tar_code.csv") and not regain:
        res = pd.read_csv(cache+"task_id_tar_code.csv")
    else:
        df = get_train_df()
        res = df[df.content_type_id == 0].groupby("task_container_id").agg({"answered_correctly": "mean"})
        res.columns = ["task_contain_tar_code"]
        res.reset_index(inplace=True)

        res.to_csv(cache+"task_id_tar_code.csv", index=False)

    return res


def get_tag_tar_code(cache=CACHE, regain=False):
    """
    :param cache:保存路径
    :param regain: 是否重新生成
    :return: tag‘s target encoding
    """
    if os.path.exists(cache+"tag_tar_code.csv") and not regain:
        res = pd.read_csv(cache+"tag_tar_code.csv")
    else:
        # load data
        question_tags = get_question_tags()
        df = get_train_df()
        df = df.loc[df.content_type_id == 0, ["content_id", "answered_correctly"]]

        # get tag target encode
        df = pd.merge(df, question_tags, on=["content_id"], how="left")
        res = df.groupby("tag").agg({"answered_correctly": "mean"})
        res.columns = ["tag_tar_code"]
        res.reset_index(inplace=True)

        # save
        res.to_csv(cache+"tag_tar_code.csv", index=False)

    return res


def get_usr_tag_feat(cache=CACHE, tail=0, span=10, regain=False):
    """
    :param cache:缓存路径
    :param tail:排除每个用户末尾tail个回答记录
    :param span:排除每个用户末尾span个作为样本的回答记录,span=0时,产出用于提交的全集特征
    :param regain:是否重新生成
    :return: 用户做某个tag的题目的数量，准确率
    """
    if os.path.exists(cache+"usr_tag_feat_for_sample_{}_{}.csv".format(tail, span)) and not regain:
        res = pd.read_csv(cache+"usr_tag_feat_for_sample_{}_{}.csv".format(tail, span))
    else:
        # load data
        question_tags = get_question_tags()
        if span == 0:
            df = get_train_df()
        else:
            df = get_sample(tail=tail, span=span)[1]
        df = df.loc[df.content_type_id == 0, ["user_id", "content_id", "answered_correctly"]]

        # get usr tag gain & count
        df = pd.merge(df, question_tags, on=["content_id"], how="left")
        res = df.groupby(["user_id", "tag"]).agg({"answered_correctly": ["count", "mean"]})
        res.columns = ["usr_tag_count", "usr_tag_accuracy"]
        res.reset_index(inplace=True)

        res.to_csv(cache+"usr_tag_feat_for_sample_{}_{}.csv".format(tail, span), index=False)

    return res


if __name__ == "__main__":
    for i in range(5):
        # get_usr_acc_feat(tail=i*10)
        # get_usr_act_time_feat(tail=i*10)
        # get_usr_qpart_acc_feat(tail=i*10)
        # get_usrxquestion_feat(tail=i*10)
        get_usr_tag_feat(tail=i*10, regain=True)

    # get_question_acc_feat(regain=True)

    # get_usr_vec_ans()

    # get_question_vec_ans()
    # get_usr_vec_lecture()
    # get_usrxquestion_feat(span=0, regain=True)
    #
    # get_usr_acc_feat(span=0, regain=True)
    # get_usr_act_time_feat(span=0, regain=True)
    # get_usr_qpart_acc_feat(span=0, regain=True)
    get_usr_tag_feat(span=0, regain=True)

    get_tag_tar_code(regain=True)




