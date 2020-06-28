import pymongo
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
import imblearn.over_sampling

dictGaus = {}
Xcol = ['director', 'playwright', 'lead', 'type', 'region', 'language', 'time']


def KFold_score(x_train_data, y_train_data):
    fold = sklearn.model_selection.KFold(n_splits=5, shuffle=True)
    c_param_range = [0.01, 0.1, 1, 10, 100]
    result_table = pd.DataFrame(index=range(len(c_param_range)), columns=['C参数', 'recall'])
    j = 0
    for c_param in c_param_range:  # C参数，正则化强度
        recall_accs = []
        result_table.loc[j, 'C参数'] = c_param
        for train_index, valid_index in fold.split(x_train_data):  # 交叉验证:训练集 验证集
            lr = sklearn.linear_model.LogisticRegression(penalty='l1', C=c_param, solver='liblinear', max_iter=100)
            # 训练，revel()矩阵向量化
            lr.fit(x_train_data.iloc[train_index, :], y_train_data.iloc[train_index, :].values.ravel())
            y_pred = lr.predict(x_train_data.iloc[valid_index, :].values)  # 预测验证集求召回率
            recall_acc = sklearn.metrics.recall_score(y_train_data.iloc[valid_index, :], y_pred)
            recall_accs.append(recall_acc)  # 统计召回率
        result_table.loc[j, 'recall'] = np.mean(recall_accs)
        j += 1
    print(result_table, '\n')
    maxval = 0
    maxid = 0  # 求最好的C参数
    for id, val in enumerate(result_table['recall']):
        if val > maxval:
            maxval = val
            maxid = id
    best_c = result_table.loc[maxid]['C参数']
    return best_c


def plot_confusion_matrix(cm, cmap=plt.cm.Blues, title='混淆矩阵'):
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.xticks(range(2))
    plt.yticks(range(2))
    plt.colorbar()
    max = 0
    for i in range(2):
        for j in range(2):
            if cm[i][j] > max:
                max = cm[i][j]
    for i in range(2):
        for j in range(2):
            if cm[i][j] > max / 2:
                color = 'white'
            else:
                color = 'black'
            font = {'family': 'Noto Sans CJK SC',
                    'style': 'italic',
                    'weight': 'normal',
                    'color': color,
                    'size': 16
                    }
            plt.text(j, i, cm[i][j], fontdict=font)


def QuanGausDist(data, key):
    global dictGaus
    dictGaus[key] = {}
    reshape = data.values.reshape(-1, 1)
    num = 0
    dic = {}
    try:
        for i in range(len(reshape)):
            reshape[i][0] = int(reshape[i][0])
            dictGaus[key][reshape[i][0]] = 0  # 记录key,value初始化0
    except:
        for i in range(len(reshape)):
            if reshape[i][0] not in dic:
                dictGaus[key][reshape[i][0]] = 0  # 记录key,value初始化0
                dic[reshape[i][0]] = num  # 编号
                reshape[i][0] = num  # 非数值型转编号
                num += 1
            else:
                reshape[i][0] = dic[reshape[i][0]]  # 非数值型转编号
    gaus = StandardScaler().fit_transform(reshape)  # 对编号进行高斯分布
    i = 0
    for k in dictGaus[key].keys():
        dictGaus[key].update({k: gaus[i][0]})  # 高斯分布值字典
        i += 1
    return gaus


if __name__ == '__main__':
    client = pymongo.MongoClient('127.0.0.1', 27017)
    mydb = client['Douban']
    sheet = mydb['Doubanmovie']
    data = pd.DataFrame(list(sheet.find()))  # 读取数据
    data.dropna(how='any')
    data = data.drop(['name', '_id'], axis=1)  # 去掉片名和上映日期 col:7+1
    scores = data['star'].values.reshape(-1, 1)
    for i in range(len(scores)):
        if float(scores[i]) >= 6.0:
            scores[i] = 1
        else:
            scores[i] = 0
    data['star'] = scores.astype(np.int64)  # 类型转换!!!
    for x in Xcol:
        data[x] = QuanGausDist(data[x], x)  # 特征量化为高斯分布
    # 划分集合
    X = data.loc[:, data.columns != 'star']  # 特征集
    Y = data.loc[:, data.columns == 'star']  # 结果集
    count_classes = pd.value_counts(data['star'])
    print(count_classes, '\n')
    # 1   916
    # 0   216
    x_train, x_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, Y, test_size=0.3, random_state=0)  # 全集合划分训练 测试
    best_c = KFold_score(x_train, y_train)
    print('C参数：', best_c, '\n')  # 得出最适C，后再进行训练
    print('训练集长度是：', len(y_train))
    print('测试集长度是：', len(y_test))
    print('总共的数量是：', len(y_train) + len(y_test), '\n')
    lr = sklearn.linear_model.LogisticRegression(penalty='l1', C=best_c, solver='liblinear', max_iter=1000)
    lr.fit(x_train, y_train.values.ravel())  # 训练
    y_pred = lr.predict(x_test.values)  # 预测
    cnf_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)  # 混淆矩阵
    recall = cnf_matrix[1][1] / (cnf_matrix[1][1] + cnf_matrix[1][0])
    print('recall:', recall)
    precison = cnf_matrix[1][1] / (cnf_matrix[1][1] + cnf_matrix[0][1])
    print('precison:', precison, '\n')
    plot_confusion_matrix((cnf_matrix))
    plt.savefig('测试集预测.png')
    plt.show()
    # oversampled
    print('#####OverSampled#####\n')
    oversampled = imblearn.over_sampling.SMOTE()
    x_train_oversampled, y_train_oversampled = oversampled.fit_sample(x_train, y_train)  # 训练集OverSamoled
    print('Over后训练集长度:', y_train_oversampled.size)
    print('总共的数量是：', len(y_train_oversampled) + len(y_test), '\n')
    best_c = KFold_score(x_train_oversampled, y_train_oversampled)
    print('C参数：', best_c, '\n')  # 得出最适C，后再进行训练
    lr = sklearn.linear_model.LogisticRegression(penalty='l1', C=best_c, solver='liblinear', max_iter=1000)
    lr.fit(x_train_oversampled, y_train_oversampled.values.ravel())  # 采样训练
    y_pred = lr.predict(x_test.values)
    cnf_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
    recall = cnf_matrix[1][1] / (cnf_matrix[1][1] + cnf_matrix[1][0])
    print('recall:', recall)
    precison = cnf_matrix[1][1] / (cnf_matrix[1][1] + cnf_matrix[0][1])
    print('precison:', precison, '\n')
    plot_confusion_matrix((cnf_matrix))
    plt.savefig('测试集over预测.png')
    plt.show()
    # 自定义预测1
    info = {'director': '曾国祥', 'playwright': '林咏琛', 'lead': '周冬雨', 'type': '剧情',
            'region': '中国大陆', 'language': '汉语普通话', 'time': 135}  # 1
    # 自定义预测2
    # info = {'director': '裘仲维', 'playwright': '吕萍', 'lead': '于斌', 'type': '奇幻',
    #         'region': '中国大陆', 'language': '汉语普通话', 'time': 84}  # 0
    ##########
    data_mod = pd.DataFrame(np.arange(0, len(Xcol)).reshape((1, len(Xcol))), columns=Xcol)
    for x in Xcol:
        data_mod[x] = dictGaus[x][info[x]]  # 查询高斯分布值
    x_mod = data_mod.loc[:, :]
    print(x_mod)
    y_pred_mod = lr.predict(x_mod.values)  # 预测
    print('自定义预测结果:', y_pred_mod)
