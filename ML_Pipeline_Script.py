import pandas as pd

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler

_pipeline_info_dict=dict({})
_eval_score_dict=dict([])

outliers_list=[]

def get_df_path(value):
    _pipeline_info_dict['path']=value

def get_drop_cols(values):
    _pipeline_info_dict['col_to_drop']=[value for value in values.split(',')]

def get_imp_tech(value):
    _pipeline_info_dict['imp_tech']=value

def get_imp_col(values):
    _pipeline_info_dict['imp_cols']=[int(value) for value in values.split(',')]

def get_cat_to_num_col(values):
    _pipeline_info_dict['cat_to_num_col']=[value for value in values.split(',')]

def get_features_label(x1,x2,y):
    _pipeline_info_dict['x_start']=x1
    _pipeline_info_dict['x_end']=x2
    _pipeline_info_dict['y']=y

def get_scaling_tech(value):
    _pipeline_info_dict['scaling_tech']=value

def get_splitting_param(*values):
    _pipeline_info_dict['splitting_param']=[value for value in values]

def get_model_name(value):
    _pipeline_info_dict['model']=value

def get_model_param(*values):
    _pipeline_info_dict['param']=[value for value in values]

#Function to identify outliers using IQR to calculate the upper and lower boundary
def DetectOutliers(df,col):
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    IQR=Q3-Q1
    
    uw=Q3+(1.5*IQR)
    lw=Q1-(1.5*IQR)
    
    for val in df[col]:
        if (val<lw or val>uw):
            outliers_list.append(val)
    return outliers_list

#Core ML pipeline
def StartPipline():
    #Reading the csv file
    dataset_path=_pipeline_info_dict['path']
    df=pd.read_csv(dataset_path)

    print(df.head(6))

    to_drop_col=_pipeline_info_dict['col_to_drop']
    
    #Dropping columns
    if to_drop_col[0] == '':
        print('No columns dropped')
    else:
        to_drop_col=[int(col) for col in to_drop_col]
        df.drop(df.columns[to_drop_col],axis=1,inplace=True)
        print('Columns dropped')

    print(df.head(6))
    
    #Handling missing values
    if _pipeline_info_dict['imp_tech']=='None':
        print('No imputation technique')
    else:
        imp_cols=_pipeline_info_dict['imp_cols']
        if _pipeline_info_dict['imp_tech']=='Drop':
            df.dropna(inplace=True)
        elif _pipeline_info_dict['imp_tech']=='Mean':
            for col in imp_cols:
                df[df.columns[col]].fillna(df[df.columns[col]].mean(),inplace=True)
        elif _pipeline_info_dict['imp_tech']=='Median':
            for col in imp_cols:
                df[df.columns[col]].fillna(df[df.columns[col]].median(),inplace=True)
        elif _pipeline_info_dict['imp_tech']=='Mode':
            for col in imp_cols:
                df[df.columns[col]].fillna(df[df.columns[col]].mode()[0],inplace=True)
        else:
            print('No imputation technique')
    
    cat_to_num_cols=_pipeline_info_dict['cat_to_num_col']

    if cat_to_num_cols[0]=='':
        print('Not mappint categorical to numberical')
    else:
        cat_to_num_cols=[int(col) for col in cat_to_num_cols]
        for cat_to_num_col in cat_to_num_cols:
            for i in range(0,df[df.columns[cat_to_num_col]].nunique()):
                df[df.columns[cat_to_num_col]].replace(df[df.columns[cat_to_num_col]].unique()[i],i,inplace=True)
    
    print(df.head(6))
    
    X=df.iloc[:,:-1].values
    Y=df.iloc[:,-1]
    
    #Data spliting - training and testing data
    test_size,random_state,shuffle=[param for param in _pipeline_info_dict['splitting_param']]
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=test_size,random_state=random_state,shuffle=shuffle)

    #Scaling the data
    if _pipeline_info_dict['scaling_tech']=='None':
        print('No scaling technique')
    else:
        if _pipeline_info_dict['scaling_tech']=='Standardization':
            scaler=StandardScaler()
        elif _pipeline_info_dict['Min-Max Scaling']=='Min-Max Scaling':
            scaler=MinMaxScaler()
        else:
            print('No scaling technique')
        x_train=scaler.fit_transform(x_train)

    #Fitting the model
    if _pipeline_info_dict['model']=='K-Nearest Neighbor':
        n_neighbors,leaf_size,weights,algorithm=[param for param in _pipeline_info_dict['param']]
        model=KNeighborsClassifier(n_neighbors=n_neighbors,leaf_size=leaf_size,weights=weights,algorithm=algorithm)
    elif _pipeline_info_dict['model']=='Support Vector Machine':
        model=SVC()
    elif _pipeline_info_dict['model']=='Logistic Regression':
        max_iterint_LogReg,random_state_LogReg,penalty_spinner,solver_spinner,multi_class_spinner=[param for param in _pipeline_info_dict['param']]
        model=LogisticRegression(max_iter=max_iterint_LogReg,random_state=random_state_LogReg,penalty=penalty_spinner,solver=solver_spinner,multi_class=multi_class_spinner)
    elif _pipeline_info_dict['model']=='Decision Tree':
        model=DecisionTreeClassifier()
    elif _pipeline_info_dict['model']=='Random Forest':
        model=RandomForestClassifier()
    else:
        print('No model selected')
    model.fit(x_train,y_train)
    print('Model fitted')

    #Prediction
    y_pred=model.predict(x_test)
    
    #Performance evaluation
    _eval_score_dict['accuracy']=str(round(accuracy_score(y_test,y_pred),3))
    _eval_score_dict['precision']=str(round(precision_score(y_test,y_pred,average='macro'),3))
    _eval_score_dict['recall']=str(round(recall_score(y_test,y_pred,average='macro'),3))
    _eval_score_dict['f1']=str(round(f1_score(y_test,y_pred,average='macro'),3))

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax,
        colorbar=False,
    )
    plt.savefig("C:\\Codes\\Python\\VisualML\\figs\\cm_plot.png", dpi=300)
    
    print('Accuracy:',_eval_score_dict['accuracy'])
    print('Precision:',_eval_score_dict['precision'])
    print('Recall:',_eval_score_dict['recall'])
    print('F1 Score:',_eval_score_dict['f1'])

    """
    df=pd.read_csv('C:\\Codes\\Datasets\\iris.csv')
    df.drop(columns=['Unnamed: 0'],inplace=True)
    
    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0,shuffle=True)
    knn=KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train.iloc[:,2:4].values,y_train.values)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_decision_regions(x_train.iloc[:,2:4].values, y_train.values, clf=knn, legend=2,ax=ax)
    plt.savefig("C:\\Codes\\Python\\VisualML\\figs\\knn.png", dpi=300)"""