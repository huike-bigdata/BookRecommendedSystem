import time

import pandas as pd
import numpy as np


class DataLoader(object):

    def getDataFrame(self, file_path, sep, encoding="utf-8", num=10000):
        """
        读取文件为pd.dataframe
        :param file_path:
        :param sep:分隔符
        :param encoding:
        :param num:读取前多少条，如果为0则返回所有
        :return:
        """
        data = pd.read_csv(file_path, sep=sep, encoding=encoding)
        if num == 0:
            return data
        else:
            # 随机选取
            # return data.sample(num, axis=0)
            return data.sample(num, axis=0)

    def processDataFrametoArray(self, dataframe, Ml="User-ID", Nl="ISBN"):
        """
        将pd.dataframe转为np.darray
        :param dataframe:
        :param Ml:用户列
        :param Nl:书籍列
        :return:返回的darray，行为每个用户给所有书籍的评分，列为用户给每个书籍的评分
        """

        user_list = list(dataframe[Ml].unique())
        # print(type(user_list))
        # print(user_list)
        # print(user_list[1])
        # 所有不重复的用户

        ISBN_list = list(dataframe[Nl].unique())
        # 所有不重复的书籍

        zero = np.zeros(shape=(len(user_list), len(ISBN_list)), dtype="int64")
        # print(zero.shape)
        # 空的array，预备放置每个用户对于每个书籍的评分
        # print("len(user_list) is: {}".format(len(user_list)))
        for u in range(len(user_list)):
            # print("u is: {}".format(u))
            for book in range(len(ISBN_list)):
                # print("book is: {}".format(book))
                # print("查找：user_list[u]: {}".format(int(user_list[u])))
                # print("查找：ISBN_list[book]: {}".format(str(ISBN_list[book])))
                zero[u, book] = self.getRating(dataframe, int(user_list[u]), str(ISBN_list[book]))
            print("{} — 读取数据进度：{}%".format(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())),
                                           round(u / len(user_list), 4) * 100))
        return zero, user_list, ISBN_list

    def getRating(self, dataframe, user_id, ISBN):
        """
        根据用户和ISBN找到用户对此书的评分
        :param user_id:
        :param ISBN:
        :return:
        """
        # print("user_id is {}, ISBN is {}".format(user_id, ISBN))
        rating = dataframe[(dataframe["User-ID"] == user_id) & (dataframe["ISBN"] == ISBN)]["Book-Rating"]
        # print("rating is: {}".format(rating))
        if len(rating) == 0:
            # 没找到评分
            return 0
        else:
            return int(rating)


if __name__ == "__main__":
    dataLoader = DataLoader()
    # num: 获取的数据条数，决定了后边处理数据的时间，以及计算时间
    ratings = dataLoader.getDataFrame("../data/BX-Book-Ratings.csv", ";", "utf-8", num=100)
    arr = dataLoader.processDataFrametoArray(ratings)
    print(arr)
    # print(ratings)
    # print(ratings["User-ID"].dtype)
    # print("评论数{}".format(+ratings["User-ID"].count()))
    # print("用户数{}".format(+ratings["User-ID"].nunique()))
    # print("用户数{}".format(len(ratings["User-ID"].unique())))
    # print("评论数{}".format(+ratings["ISBN"].count()))
    # print("书籍数{}".format(+ratings["ISBN"].nunique()))
    # print("评论数{}".format(+ratings["Book-Rating"].count()))
    # print("不重复的的评分{}".format(+ratings["Book-Rating"].nunique()))
    # print(ratings.iloc[:,1])
    # print("书籍数{}".format(ratings["ISBN"].unique().shape().Length))
    # print(ratings[(ratings["User-ID"] == 276725) & (ratings["ISBN"] == "0155061224")]["Book-Rating"])
    # print(ratings[ratings["ISBN"] == "052165615X"])
    # print(int(dataLoader.getRating(ratings, 276725, "034545104X")))
