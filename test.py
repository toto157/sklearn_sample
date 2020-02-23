from sklearn.datasets import fetch_20newsgroups,load_boston,load_diabetes
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression,Ridge,SGDRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
def naviebayes():
    """朴素贝叶斯文本分类"""
    news = fetch_20newsgroups(1)
    print(news)
    x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.3)
    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)
    mlt = MultinomialNB(alpha=1.0)
    mlt.fit(x_train,y_train)
    y_predict = mlt.predict(x_test)
    print(y_predict)
    print(mlt.score(x_test,y_test))
    print(classification_report(y_test,y_predict,target_names=news.target_names))

# 决策树随机森林预测titanic生还者
def decision():
    titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

    # print(titanic)

    x = titanic[['pclass','age','sex']]

    y = titanic['survived']

    # 去age中nan 以mean填充
    x.iloc[:,1].fillna(x.iloc[:,1].mean(),inplace=True)

    # 分割数据
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

    # 数据特征处理 one-hot编码
    dict = DictVectorizer(sparse=False)

    x_train = dict.fit_transform(x_train.to_dict(orient='records'))

    x_test = dict.transform(x_test.to_dict(orient='records'))

    # 随机森林 使用所有cpu运行
    rfcls = RandomForestClassifier(n_jobs=-1)

    # 调试超参
    p = {'n_estimators':range(300,900,50),'max_depth':range(1,9,1)}

    # 网格 交叉验证
    gscv = GridSearchCV(estimator=rfcls,param_grid=p,cv=3)

    gscv.fit(x_train,y_train)

    # 查看准确率
    print('预测准确率',gscv.score(x_test,y_test))

    # 查看随机森林树数量 深度
    print(gscv.best_params_)


# 线性回归预测房子价格
def linears():
    """根据官方建议 数据量小于10万 以下情况用岭回归准确率比较合适 大于10万用SGDRegressor比较合适 故在此采用Ridge"""
    lb = load_boston()
    # print(lb)

    # 分割数据
    x_train,x_test,y_train,y_test = train_test_split(lb.data,lb.target,test_size=0.3)

    # x_train.dropna(replace)

    # 标准化处理特征值
    stdx = StandardScaler()

    x_train = stdx.fit_transform(x_train)
    x_test = stdx.transform(x_test)

    # 标准化处理目标值
    stdy = StandardScaler()

    # 一维数组变二维
    y_train = stdy.fit_transform(pd.DataFrame(y_train))
    y_test = stdy.transform(pd.DataFrame(y_test))

    # 岭回归
    rd = Ridge(alpha=1.0)

    rd.fit(x_train,y_train)

    # 预测房价 将标准化数据转换回原始数据
    y_rd_predict = stdy.inverse_transform(rd.predict(x_test))

    # joblib.load('bst_rd.pkl')

    # 保存模型
    joblib.dump(y_rd_predict,"bst_rd.pkl")


    avrg = str(round(mean_squared_error(stdy.inverse_transform(y_test),y_rd_predict),2))
    predict = pd.DataFrame(y_rd_predict)
    fect = pd.DataFrame(y_test)+predict

    p1 = plt.plot(predict)
    p2 = plt.plot(fect)
    plt.bar( 0.35,predict, label='predict')
    plt.bar( 0.35,fect, label='fect')
    plt.ylabel('Mean squared error: '+avrg)
    plt.title('Boston predict and fect.')
    plt.legend((p1[0],p2[0],),('Predict','Fect'))
    plt.savefig('./Boston.png')

    plt.show()




if __name__ == '__main__':
    # 调用决策树随机森林预测titanic生还者:
    # decision()
    # 线性回归预测房屋价格
    linears()