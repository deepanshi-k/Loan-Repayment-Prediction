import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier

df = open(r"C:\Users\deepa\OneDrive\Desktop\Project sem 4\loan_data.csv")
df = pd.read_csv(df)

df = df.drop(['days.with.cr.line', 'revol.bal', 'revol.util', 'delinq.2yrs', 'pub.rec','credit.policy'], axis = 1)
df['purpose']=LabelEncoder().fit_transform(df['purpose'])

X = df.iloc[:,0:7].values
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

R_model = RandomForestClassifier(n_estimators = 1000)
R_model.fit(X_train, y_train)

pickle.dump(R_model,open("model.pkl","wb"))