import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
s=pd.read_csv('titanic_train.csv')
df=pd.DataFrame(s)
# print(df)
# df.drop('Cabin',axis=1,inplace=True)

# def inputal(colm):
#        p=colm[0]
#        A=colm[1]
#
#        if pd.isnull(A):
#               if p==1:
#                      return 37
#               if p==2:
#                      return 29
#               else:
#                      return 24
#        else:
#               return A
#
# df["Age"]=df[["Pclass","Age"]].apply(inputal,axis=1)
# df.dropna(axis=0,inplace=True)
# sns.heatmap(df.isnull())
# print(df.drop.loc['Age'])
# plt.show()
#
# df2=df['Pclass'].mean()
# k=(df.fillna(df2))
# print(k.head())
# sns.boxplot(x="Pclass",y="Age")
# plt.show()


x=df[['Pclass','Parch','Fare']]
y=df['Survived']

x_trained, x_test , y_trained, y_test = train_test_split(x,y,test_size=.4,random_state=101)
# print(x_test)
lm = LogisticRegression()
lm.fit(x_trained,y_trained)#method calling
pp=(lm.coef_)

l=pd.DataFrame(pp,index=['Pclass','Parch','Fare'],columns=["coef"])
print(l)


# sns.countplot(x="Survived",data=df)
# plt.show()



# sns.heatmap(df.corr(),annot=True)
# plt.show()