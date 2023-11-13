#讀檔
import pandas as pd
data = pd.read_csv("character-deaths.csv")

#刪掉 Book of Death、 Death Chapter
data = data.drop(['Book of Death','Death Chapter'],axis=1)
data

#利用 Death Year 判斷 IsDeath
data['IsDeath'] = data['Death Year'].isnull().map(lambda x : 0 if x == True else 1)
data

#刪掉 Death Year、Book Intro Chapter
data = data.drop(['Death Year','Book Intro Chapter'],axis=1)
data

#檢查變數是否還有遺失值
print(data.isnull().sum(axis = 0)/data.shape[0])

#刪掉姓名、Allegiances轉 dummy 
data = data.drop(['Name'],axis=1)
data = pd.get_dummies(data)
data

#亂數拆成訓練集(75%)與測試集(25%) 
from sklearn.model_selection import train_test_split

X = data.drop('IsDeath',axis=1)
y = data['IsDeath']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)

#使用決策樹演算法
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=3) #深度為3
dtree.fit(X_train,y_train)

#做出Confusion Matrix，並計算Precision, Recall, Accuracy
predictions = dtree.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predictions)
print("Confusion Matrix:","\n",cm)

precision = round(cm[1][1]/(cm[0][1]+cm[1][1]),3)
recall = round(cm[1][1]/(cm[1][0]+cm[1][1]),3)
accuracy = round((cm[0][0]+cm[1][1])/sum(sum(cm)),3)

print("----------------------")
print("precision:",precision)
print("recall:",recall)
print("accuracy:",accuracy)


#畫決策樹
import graphviz
from sklearn import tree
from graphviz import Graph


# DOT data
dot_data = tree.export_graphviz(dtree, out_file=None, 
                                feature_names=list(X_train.columns),  
                                class_names=["0","1"],
                                filled=True)

# Draw and save graph 
graph = graphviz.Source(dot_data, format="png") 
graph.render('graph.gv',format="jpg")
graph