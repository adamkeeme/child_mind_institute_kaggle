from read_data import data_load
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB

datasets = ['2-2_train_iterative_imputed',
'3-3_train_simple_imputed',
'3-2_train_knn_imputed',
'3-1_train_iterative_imputed',
'2-1_train_simple_imputed',
'2-3_train_knn_imputed',
'1_COMPLETE_train_clean_without_fitness_FGC_no_nulls']

def convert_onehot(dataset):
    # retrieve object features
    dataset = dataset.drop(columns=['id'])
    object_columns = dataset.select_dtypes(include=['object']).columns
    # generate one_hot
    df_onehot = dataset.drop(columns=object_columns)
    # convert bool into float
    #print(df_onehot.shape)
    

    return df_onehot

def train_gb(X_train, X_test, y_train, y_test):
    gbdt = GradientBoostingClassifier(
        n_estimators=100,  
        learning_rate=0.1, 
        max_depth=3,       
        random_state=42
    )
    gbdt.fit(X_train, y_train)
    y_pred = gbdt.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

def naive_bayes(X_train, X_test, y_train, y_test):
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))


alldatasets = data_load()
for data in datasets:
    dataset = alldatasets[data]
    # Handle non-numeric columns with one-hot encoding
    X_train = convert_onehot(dataset['X_train'])
    X_test = convert_onehot(dataset['X_test'])
    y_train = dataset['y_train']
    y_test = dataset['y_test']
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(f'training for {data}')
    #train_gb(X_train, X_test, y_train, y_test)
    naive_bayes(X_train, X_test, y_train, y_test)
'''
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
'''
'''for data in datasets:
    print(f'Training for {data}')
    dataset = alldatasets[data]
    X_train = convert_onehot(dataset['X_train'])
    X_test = convert_onehot(dataset['X_test'])
    y_train = dataset['y_train']
    y_test = dataset['y_test']
    

    train_gb(X_train, X_test, y_train, y_test)'''




