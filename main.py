import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image

st.set_page_config(page_title="Heart Disease Prediction", layout="wide", page_icon="❤")
st.write("""
# Heart Disease Prediction using Machine Learning

This dashboard created by : [Izzuddin Al Qossam](https://id.linkedin.com/in/izzuddin-al-qossam-04970589). Dataset provided by : [UCIML](https://archive.ics.uci.edu/dataset/45/heart+disease)
""")
add_selectitem = st.sidebar.selectbox("Navigation", ("Overview", "Data Preprocessing", "Model Analysis", "Prediction", "About"))

def overview():
    st.header('Dataset Overview')
    st.subheader('Information')
    st.write('''
            This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them.  In particular, the Cleveland database is the only one that has been used by ML researchers to date.  The "goal" field refers to the presence of heart disease in the patient.
             ''')
    data={
    'Variable Name': ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target'],
    'Label': ['Feature','Feature','Feature','Feature','Feature','Feature','Feature','Feature','Feature','Feature','Feature','Feature','Feature','Target'],
    'Data Type': ['Numerical','Categorical','Categorical','Numerical','Numerical','Categorical','Categorical','Numerical','Categorical','Numerical','Categorical','Categorical','Categorical','Categorical'],
    'Missing Values': ['No','No','No','No','No','No','No','No','No','No','No','Yes','Yes','No'],
    'Description': ['The age of the patient (in years).','Gender (1 = male, 0 = female).','Type of chest pain (angina, atypical angina, non-anginal, asymptomatic).','Blood pressure at rest (in mm Hg).','Serum cholesterol (in mg/dl).','Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).','Electrocardiography result (0, 1, 2).','Maximum heart rate achieved.','Exercise-induced angina (1 = yes, 0 = no).','Exercise-induced ST depression relative to rest.','ST segment slope (upsloping, flat, downsloping).','Number of major blood vessels colored (0-3).','Thalassemia status (normal, fixed defect, reversible defect).','Heart disease diagnosis (0 = none, 1 = present).']
    }
    df=pd.DataFrame(data)
    st.dataframe(df, hide_index=True)

    st.subheader('Data Sneak-Peek')
    st.write('Data shape (Rows, Columns)')
    st.code('1025 rows × 14 columns')
    st.write('First five data')
    data={
    'age': [52,53,70,61,62],
    'sex': [1,1,1,1,0],
    'cp': [0,0,0,0,0],
    'trestbps': [125,140,145,148,138],
    'chol': [212,203,174,203,294],
    'fbs': [0,1,0,0,1],
    'restecg': [1,0,1,1,1],
    'thalach': [168,155,125,161,106],
    'exang': [0,1,1,0,0],
    'oldpeak': [1,3.1,2.6,0,1.9],
    'slope': [2,0,0,2,1],
    'ca': [2,0,0,1,3],
    'thal': [3,3,3,3,2],
    'target': [0,0,0,0,0]
    }
    df=pd.DataFrame(data)
    st.dataframe(df, hide_index=True)

    st.write('Last five data')
    data={
    'age': [59,60,47,50,54],
    'sex': [1,1,1,0,1],
    'cp': [1,0,0,0,0],
    'trestbps': [140,125,110,110,120],
    'chol': [221,258,275,254,188],
    'fbs': [0,0,0,0,0],
    'restecg': [1,0,0,0,1],
    'thalach': [164,141,118,159,113],
    'exang': [1,1,1,0,0],
    'oldpeak': [0,2.8,1,0,1.4],
    'slope': [2,1,1,2,1],
    'ca': [0,1,1,0,1],
    'thal': [2,3,2,2,3],
    'target': [1,0,0,1,0]
    }
    df=pd.DataFrame(data)
    st.dataframe(df, hide_index=True)
   
def dataprep():
    st.header('Data Preprocessing and Exploratory Data Analysis')

    #'''Data Deskriptif'''
    st.subheader('Descriptive Data')
    data={'' : ['count','mean','std','min','25%','50%','75%','max'],
    'age': [1025,54.434146,9.072290,29,48,56,61,77],
    'sex': [1025,0.695610,0.460373,0,0,1,1,1],
    'cp': [1025,0.942439,1.029641,0,0,1,2,3],
    'trestbps': [1025,131.611707,17.516718,94,120,130,140,200],
    'chol': [1025,246,51.592510,126,211,240,275,564],
    'fbs': [1025,0.149268,0.356527,0,0,0,0,1],
    'restecg': [1025,0.529756,0.527878,0,0,1,1,2],
    'thalach': [1025,149.114146,23.005724,71,132,152,166,202],
    'exang': [1025,0.336585,0.472772,0,0,0,1,1],
    'oldpeak': [1025,1.071512,1.175053,0,0,0.8,1.8,6.2],
    'slope': [1025,1.385366,0.617755,0,1,1,2,2],
    'ca': [1025,0.754146,1.030798,0,0,0,1,4],
    'thal': [1025,2.323902,0.620660,0,2,2,3,3],
    'target': [1025,0.513171,0.500070,0,0,1,1,1]
    }
    df=pd.DataFrame(data)
    st.dataframe(df, hide_index=True, width=1050)

    #'''Tipe Data dan Keunikan Data'''
    st.subheader('Data Type and Data Uniqueness')
    # Data Type and Data Uniqueness DataFrames
    data1={'' : ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target'],
    'type' : ['int64','int64','int64','int64','int64','int64','int64','int64','int64','float64','int64','int64','int64','int64']
    }
    data2={'' : ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target'],
    'unique value': [41,2,4,49,152,2,3,91,2,40,3,5,4,2]
    }
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    # Create two columns
    col1, col2 = st.columns([1,1])  
    # Display tables in separate columns
    col1.dataframe(df1, hide_index=True, width=500)
    col2.dataframe(df2, hide_index=True, width=500)
    col1.write('''Categorical data ('sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'ca', and 'target') were labeled by converting the initial data type of integer to the data type of object.''')
    col2.write('Several categorical columns exceed the unique value limits defined by UCI, suggesting the possibility of human error.')
    
    st.subheader('Data Labelling')
    st.code('''
    # Labelling categorical data
    data['sex'] = data['sex'].replace({1: 'Male',
                                    0: 'Female'})
    data['cp'] = data['cp'].replace({0: 'typical angina',
                                    1: 'atypical angina',
                                    2: 'non-anginal pain',
                                    3: 'asymtomatic'})
    data['fbs'] = data['fbs'].replace({0: 'No',
                                    1: 'Yes'})
    data['restecg'] = data['restecg'].replace({0: 'probable or definite left ventricular hypertrophy',
                                            1:'normal',
                                            2: 'ST-T Wave abnormal'})
    data['exang'] = data['exang'].replace({0: 'No',
                                        1: 'Yes'})
    data['slope'] = data['slope'].replace({0: 'downsloping',
                                        1: 'flat',
                                        2: 'upsloping'})
    data['thal'] = data['thal'].replace({1: 'normal',
                                        2: 'fixed defect',
                                        3: 'reversable defect'})
    data['ca'] = data['ca'].replace({0: 'Number of major vessels: 0',
                                    1: 'Number of major vessels: 1',
                                    2: 'Number of major vessels: 2',
                                    3: 'Number of major vessels: 3'})
    data['target'] = data['target'].replace({0: 'No disease',
                                            1: 'Disease'})
    ''')
    data={
    'age': [52,53,70,61,62,58,58,55,46,54],
    'sex': ['Male','Male','Male','Male','Female','Female','Male','Male','Male','Male'],
    'cp': ['typical angina','typical angina','typical angina','typical angina','typical angina','typical angina','typical angina','typical angina','typical angina','typical angina'],
    'trestbps': [125,140,145,148,138,100,114,160,120,122],
    'chol': [212,203,174,203,294,248,318,289,249,286],
    'fbs': ['No','Yes','No','No','Yes','No','No','No','No','No',],
    'restecg': ['normal','probable or definite left ventricular hypertrophy','normal','normal','normal','probable or definite left ventricular hypertrophy','ST-T Wave abnormal','probable or definite left ventricular hypertrophy','probable or definite left ventricular hypertrophy','probable or definite left ventricular hypertrophy',],
    'thalach': [168,155,125,161,106,122,140,145,144,116],
    'exang': ['No','Yes','Yes','No','No','No','No','Yes','No','Yes',],
    'oldpeak': [1,3.1,2.6,0,1.9,1,4.4,0.8,0.8,3.2],
    'slope': ['upsloping','downsloping','downsloping','upsloping','flat','flat','downsloping','flat','upsloping','flat'],
    'ca': ['Number of major vessels: 2','Number of major vessels: 0','Number of major vessels: 0','Number of major vessels: 1','Number of major vessels: 3','Number of major vessels: 0','Number of major vessels: 3','Number of major vessels: 1','Number of major vessels: 0','Number of major vessels: 2'],
    'thal': ['reversable defect','reversable defect','reversable defect','reversable defect','fixed defect','fixed defect','normal','reversable defect','reversable defect','fixed defect'],
    'target': ['No disease','No disease','No disease','No disease','No disease','Disease','No disease','No disease','No disease','No disease']
    }
    df=pd.DataFrame(data)
    st.dataframe(df, hide_index=True, width=1050)


    st.subheader('Converting Invalid Data to NaN')
    '''
    The data used still had input errors in the category column. In the **'ca'** column,
    there was an invalid input value of 4, while in the **'thal'** column,
    there was an incorrect input value of 0.
    These errors were corrected by changing the values to NaN.


    '''
    st.code('''
    # Showing the sum values in 'ca' column
    data['ca'].value_counts()
            
    # Convert the 'ca' value of '4' to NaN
    data.loc[data['ca']==4, 'ca'] = np.nan
            
    # # Showing the sum values in 'thal' column
    data['thal'].value_counts()
    
    # Convert the 'thal' value of '0' to NaN
    data.loc[data['thal']==0, 'thal'] = np.nan
    ''')
    st.write('''Converting **'ca'** invalid data to NaN''')
    # Comparing ca DataFrames
    data1={'ca' : ['Number of major vessels: 0','Number of major vessels: 1','Number of major vessels: 2','Number of major vessels: 3','4'],
    'count' : [578,226,134,69,18]
    }
    data2={'ca' : ['Number of major vessels: 0','Number of major vessels: 1','Number of major vessels: 2','Number of major vessels: 3'],
    'count' : [578,226,134,69]
    }
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    # Create two columns
    col1, col2 = st.columns([1,1])  
    # Display tables in separate columns
    col1.dataframe(df1, hide_index=True, width=500)
    col2.dataframe(df2, hide_index=True, width=500)


    st.write('''Converting **'thal'** invalid data to NaN''')
    # Comparing thal DataFrames
    data1={'thal' : ['fixed defect','reversable defect','normal','0'],
    'count' : [544,410,64,7]
    }
    data2={'thal' : ['fixed defect','reversable defect','normal'],
    'count' : [544,410,64]
    }
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    # Create two columns
    col1, col2 = st.columns([1,1])  
    # Display tables in separate columns
    col1.dataframe(df1, hide_index=True, width=500)
    col2.dataframe(df2, hide_index=True, width=500)


    # '''Missing Values'''
    # Comparing Missing Values DataFrames
    data1={'' : ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target'],
    'sum of missing value' : [0,0,0,0,0,0,0,0,0,0,0,18,7,0]
    }
    data2={'' : ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target'],
    'sum of missing value': [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    }
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    # Create two columns
    col1, col2 = st.columns([1,1])  
    # Display tables in separate columns
    col1.subheader('Missing Values')
    col1.code('''
    # Check missing values
    data.isnull().sum()
    ''')
    col1.dataframe(df1, hide_index=True, width=500)
    col1.write('''Missing values (NaN) in the 'ca' and 'thal' columns will be filled using mode because both columns are categorical.
               In addition, duplicate data in the dataframe will be removed to maintain the uniqueness of each entry.
    ''')
    col2.subheader('Filling Missing Values')
    col2.code('''
    # Filling the 'ca' column with mode
    modus_ca = data['ca'].mode()[0]
    data['ca'] = data['ca'].fillna(modus_ca)

    # Filling the 'thal' column with mode
    modus_thal = data['thal'].mode()[0]
    data['thal'] = data['thal'].fillna(modus_thal)
    ''')
    col2.dataframe(df2, hide_index=True, width=500)


    # '''Handling Duplicated Data'''
    st.subheader('Handling Duplicated Data')
    st.code('''
    # Showing duplicate data in a table
    data[data.duplicated()]
            
    # Dropping duplicate data but keeping first data
    data.drop_duplicates(keep='first', inplace=True)

    # Rechecking duplicate data
    duplicate_check=data.duplicated().any()

    print("Is there still any duplicate data?", duplicate_check)
    data[data.duplicated()]
    ''')

    st.subheader('Checking Outlier Data')
    '''
    Knowing data outliers is important because outliers can affect the results of an analysis or model. Outliers are data that are significantly different from most other data. The method used is IQR (Interquartile Range).
    '''
    st.latex(r''' Lower Bound = Q1 - 1.5 × IQR ''')
    st.latex(r''' Upper Bound = Q3 + 1.5 × IQR ''')
    st.latex(r''' IQR = Q3 - Q1 ''')
    '''where'''
    st.write(''' Q1 = First quartile (25% of data) ''')
    st.write(''' Q3 = Third quartile (75% of data) ''')
    
    '''
    Data that falls outside Lower Bound and Upper Bound are outliers.
    '''
    st.code('''
    # Showing outliers using boxplot
    data.plot(kind = 'box', subplots = True, layout = (2,7), sharex = False, sharey = False, figsize = (20, 10), color = 'k')
    plt.show()
    ''')
    img = Image.open("outlier.png")
    st.image(img)
    st.caption('Outliers Data')
    
    st.code('''
    # Knowing outliers data by defining the function
    continous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']  
    def outliers(data_out, drop = False):
        for each_feature in data_out.columns:
            feature_data = data_out[each_feature]
            Q1 = np.percentile(feature_data, 25.) 
            Q3 = np.percentile(feature_data, 75.) 
            IQR = Q3-Q1 
            outlier_step = IQR * 1.5 
            outliers = feature_data[~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))].index.tolist()  
            if not drop:
                print('For the feature {}, Num of Outliers is {}'.format(each_feature, len(outliers)))
            if drop:
                data.drop(outliers, inplace = True, errors = 'ignore')
                print('Outliers from {} feature removed'.format(each_feature))

    # Showing ouliers data
    outliers(data[continous_features])
    
    # Dropping outliers permanently
    outliers(data[continous_features], drop=True)
    
    # Checking the change
    outliers(data[continous_features])
            
    # Checking outliers using boxplot
    data.plot(kind = 'box', subplots = True, layout = (2,7), sharex = False, sharey = False, figsize = (20, 10), color = 'k')
    plt.show()
    ''')
    img = Image.open("outlier1.png")
    st.image(img)
    st.caption('Removed Outliers Data')


    st.subheader('Imbalanced Data')
    '''
    Data balance identification is performed to assess the uniformity of the dataset, ensuring that it does not lead to a biased model.
    '''
    st.code('''
    # Checking imbalanced data
    ax = sns.countplot(data['target'])
    for label in ax.containers:
        ax.bar_label(label)
    sns.despine()
    plt.show()
    ''')
    img = Image.open("balance.png")
    st.image(img)
    st.caption('Distribution of Target Data')
    '''
    The percentage of “Disease” data is known to be 55.83% and “No disease” data is 44.16%. From the distribution of the two target data which is not too far away, it can be said that this data is quite balanced.
    '''

    st.subheader('Visualization of Numerical Variable Distribution')
    st.code('''
    plt.figure(figsize=(16,8))
    for index,column in enumerate(numerical_col):
        plt.subplot(2,3,index+1)
        sns.histplot(data=numerical_col,x=column,kde=True)
        plt.xticks(rotation = 90)
    plt.tight_layout(pad = 1.0)
    plt.show()
    ''')
    img = Image.open("numerical.png")
    st.image(img)
    st.caption('Distribution of Numerical Variable')

    st.subheader('Visualization of Categorical Variable Distribution')
    st.code('''
    plt.figure(figsize=(12,12))
    for index, column in enumerate(categorical_col):
        plt.subplot(4, 3, index+1)
        sns.countplot(data=categorical_col,x=column, hue='target', palette='magma')
        plt.xlabel(column.upper(),fontsize=14)
        plt.ylabel("count", fontsize=14)
    plt.tight_layout(pad = 1.0)
    plt.show()
    ''')
    img = Image.open("categorical.png")
    st.image(img)
    st.caption('Distribution of Categorical Variable')

    st.subheader('Correlations between Data Variables')
    '''
    Correlation between data variables is employed to find important features. A positive correlation with a particular variable means that the higher the variable, the higher the probability of heart disease, while a negative correlation means that the lower the value of the variable, the higher the probability of heart disease.
    '''
    st.code('''
    plt.figure(figsize=(20,20))
    cor = df.corr()
    sns.heatmap(cor,annot=True, linewidth=.5, cmap="magma")
    plt.show()
    ''')
    img = Image.open("korelasi.png")
    st.image(img)
    st.caption('Correlation Values between Data Variables')
    '''
    
    '''
    # Creating two column text
    data={'' : ['ca','oldpeak','exang','thal','sex','age','fbs','restecg','slope','cp','thalach','target'],
    'correlation value' : [-0.456989,-0.434108,-0.431599,-0.370759,-0.318896,-0.222416,-0.027210,0.171453,0.326473,0.416319,0.422559,1.000000]
    }
    df = pd.DataFrame(data)
    # Create two columns
    col1, col2 = st.columns([1,1])  
    col1.dataframe(df, hide_index=True, width=500)
    col2.write('''
    1. 'cp', 'thalach', and 'slope' are strongly positively correlated with 'target'.
    2. 'oldpeak', 'exang', 'ca', 'thal', 'sex', and 'age' were strongly correlated with 'target'.
    3. 'fbs', 'chol', 'trestbps', and 'restecg' have weak correlation with 'target'.
    
    The features selected for analysis are: 'cp', 'thalach', 'slope', 'oldpeak', 'exang', 'ca', 'thal', 'sex', and 'age' to be further analyzed together with 'target'.
    ''')

    st.subheader('Data Preparation before Applied into Algorithm')
    '''
    **Separation of Independent and Dependent Variables**
    
    The separation of independent and dependent variables is employed
    to separate the feature column as the independent variable
    from the target column as the dependent variable,
    which is a requirement for analysis. 
    '''
    st.code('''
    X = df.drop('target', axis=1)
    y = df.target
    ''')

    '''
    **Data Scaling**
    
    Data scaling is the process of adjusting the values in a dataset
    to fall within the same range. This is important to ensure that
    large-scale features do not dominate the model results over
    small-scale features. Scaling also helps the algorithm work
    more efficiently, speeding up the optimization process and
    making all features equally weighted. 
    '''
    st.code('''
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pd.DataFrame(X_scaled)
    ''')
    col1, col2 = st.columns([1,2])
    col1.write('**Normal Data**')
    col2.write('**Scaled Data**')
    img1 = Image.open("normal.png")
    img2 = Image.open("scaled.png")
    col1.image(img1)
    col2.image(img2)
    '''
    
    '''

    '''
    
    **Data Splitting**
    
    Data splitting is also used to separate the training
    and testing datasets. The dataset ratio is 80:20 with
    a random state of 42 using data that has been scaled. 
    '''
    st.code('''
    # Defining the data splitting
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    ''')
    


def model():
    st.header('Heart Disease Prediction Modelling')
    '''
    **Preparing the libraries**
    '''
    st.code('''
    # Libraries
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
    from sklearn.model_selection import GridSearchCV
    ''')

    ##########    Logistic Regression      ##########
    st.header('Logistic Regression')
    '''
    The data was analyzed using **Logistic Regression** algorithm
    and the accuracy results were compared with the accuracy
    results after **Hypermeter Tuning**.
    '''
    st.code('''
    # Logistic regression
    clf = LogisticRegression()
    # train the classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # printing the test accuracy
    print("The test accuracy score of Logistric Regression Classifier is ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    ''')
    st.code('''
    # Logistic regression classifier with hyperparameter
    clf = LogisticRegression()
    param_grid = {
        'max_iter': [50, 100, 150, 200],
        'multi_class': ['auto', 'ovr', 'multinomial'],
        'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    }
    gs1 = GridSearchCV(
            estimator= clf,
            param_grid = param_grid, 
            cv=5, 
            n_jobs=-1, 
            scoring='roc_auc'
        )
    fit_clf_lg = gs1.fit(X_train, y_train)
    print(fit_clf_lg.best_score_)
    print(fit_clf_lg.best_params_)
    y_pred = fit_clf_lg.predict(X_test)
    print(classification_report(y_test, y_pred))
    ''')
    col1, col2 = st.columns([1,1])
    col1.write('**Before Tuning**')
    col2.write('**After Tuning**')
    img1 = Image.open("lr1.png")
    img2 = Image.open("lr2.png")
    col1.image(img1, width=500)
    col2.image(img2, width=500)
    '''
    Comparison of accuracy results before and after
    tuning shows no change. It can be concluded that
    the parameters before and after tuning are the
    best for this algorithm. 
    '''

    ##########    Decision Tree      ##########
    st.header('Decision Tree')
    '''
    The data was analyzed using **Decision Tree** algorithm
    and the accuracy results were compared with the accuracy
    results after **Hypermeter Tuning**.
    '''
    st.code('''
    # Decision Tree classifier
    clf = DecisionTreeClassifier()
    # train the classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # printing the test accuracy
    print("The test accuracy score of Decision Tree Classifier is ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    ''')
    st.code('''
    # Decision Tree classifier with hyperparameter
    clf = DecisionTreeClassifier()
    param_grid = {'min_samples_leaf': [1,3,5,7,9,10], 
                'max_depth': [1,3,5,7,9], 
                'criterion': ['gini', 'entropy', 'log_loss']}
    gs1 = GridSearchCV(
            estimator=clf,
            param_grid = param_grid, 
            cv=5, 
            n_jobs=-1, 
            scoring='roc_auc'
        )
    fit_clf_dt = gs1.fit(X_train, y_train)
    print(fit_clf_dt.best_score_)
    print(fit_clf_dt.best_params_)
    y_pred = fit_clf_dt.predict(X_test)
    print(classification_report(y_test, y_pred))
    ''')
    col1, col2 = st.columns([1,1])
    col1.write('**Before Tuning**')
    col2.write('**After Tuning**')
    img1 = Image.open("dt1.png")
    img2 = Image.open("dt2.png")
    col1.image(img1, width=500)
    col2.image(img2, width=500)
    '''
    There was a significant increase in accuracy
    from 72% to 81% after tuning. The f1-score value
    also increased significantly.
    '''

    ##########    Random Forest      ##########
    st.header('Random Forest')
    '''
    The data was analyzed using **Random Forest** algorithm
    and the accuracy results were compared with the accuracy
    results after **Hypermeter Tuning**.
    '''
    st.code('''
    # Random Forest classifier
    clf = RandomForestClassifier()
    # train the classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # printing the test accuracy
    print("The test accuracy score of Random Forest Classifier is ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    ''')
    st.code('''
    # Random Forest classifier with hyperparameter
    clf = RandomForestClassifier()
    param_grid = {'n_estimators': [50, 80, 90], 
                'max_depth': [2, 4, 6, 8, 10], 
                'criterion': ['gini', 'entropy', 'log_loss']}
    gs1 = GridSearchCV(
            estimator=clf,
            param_grid = param_grid, 
            cv=5, 
            n_jobs=-1, 
            scoring='roc_auc'
        )
    fit_clf_rf = gs1.fit(X_train, y_train)
    print(fit_clf_rf.best_score_)
    print(fit_clf_rf.best_params_)
    y_pred = fit_clf_rf.predict(X_test)
    print(classification_report(y_test, y_pred))
    ''')
    col1, col2 = st.columns([1,1])
    col1.write('**Before Tuning**')
    col2.write('**After Tuning**')
    img1 = Image.open("rf1.png")
    img2 = Image.open("rf2.png")
    col1.image(img1, width=500)
    col2.image(img2, width=500)
    '''
    There was a high increase in accuracy from
    84% to 86% after tuning. The f1-score value
    also experienced a high increase.
    '''

    ##########    Multi Layer Perceptron      ##########
    st.header('Multi Layer Perceptron')
    '''
    The data was analyzed using **Multi Layer Perceptron** algorithm
    and the accuracy results were compared with the accuracy
    results after **Hypermeter Tuning**.
    '''
    st.code('''
    # MLP Classifier
    clf = MLPClassifier()
    # train the classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # printing the test accuracy
    print("The test accuracy score of MLP Classifier is ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    ''')
    st.code('''
    # MLP Classifier with hyperparameter
    clf = MLPClassifier()
    param_grid1 = {'hidden_layer_sizes': [8, 10, 20, 50, 100, 150, 200], 
                'activation': ['identity','logistic','tanh','relu'], 
                'solver': ['sgd','lbfgs','adam']}
    gs1 = GridSearchCV(
            estimator=clf,
            param_grid = param_grid1, 
            cv=5, 
            n_jobs=-1, 
            scoring='roc_auc'
        )
    fit_clf_mlp = gs1.fit(X_train, y_train)
    print(fit_clf_mlp.best_score_)
    print(fit_clf_mlp.best_params_)
    y_pred = fit_clf_mlp.predict(X_test)
    print(classification_report(y_test, y_pred))
    ''')
    col1, col2 = st.columns([1,1])
    col1.write('**Before Tuning**')
    col2.write('**After Tuning**')
    img1 = Image.open("mlp1.png")
    img2 = Image.open("mlp2.png")
    col1.image(img1, width=500)
    col2.image(img2, width=500)
    '''
    Comparison of accuracy results before and after
    tuning shows no change. It can be concluded that
    the parameters before and after tuning are the
    best for this algorithm. 
    '''

    ##########    Model Comparison      ##########
    st.header('Model Comparison')
    data={'Model':['Logistic Regression', 'Random Forest', 'Decision Tree', 'MLP'],
       'Before Tuning':[0.84,0.84,0.72,0.84],
       'After Tuning':[0.84,0.86,0.81,0.84]}
    df=pd.DataFrame(data)
    st.dataframe(df, hide_index=True, width=1050)

    ##########    ROC-AUC Analysis      ##########
    st.header('ROC-AUC Analysis')
    '''
    **What is the ROC-AUC**

    The ROC-AUC, Receiver Operating Characteristic - Area Under Curve,
    is a graphical representation of the performance of a binary classification
    model at various classification thresholds. It is commonly used in machine
    learning to assess the ability of a model to distinguish between two classes,
    typically the positive class (e.g., presence of a disease) and the negative
    class (e.g., absence of a disease).
    
    
    **Receiver Operating Characteristics (ROC)**

    ROC stands for Receiver Operating Characteristics, and the ROC curve is
    the graphical representation of the effectiveness of the binary
    classification model. It plots the True Positive Rate (TPR) vs
    the False Positive Rate (FPR) at different classification thresholds.

    
    **Area Under Curve (AUC)**

    AUC stands for the Area Under the Curve, and the AUC represents the area
    under the ROC curve. It measures the overall performance of the binary
    classification model. As both TPR and FPR range between 0 to 1, So, the
    area will always lie between 0 and 1, and A greater value of AUC denotes
    better model performance. Our main goal is to maximize this area in order
    to have the highest TPR and lowest FPR at the given threshold. The AUC
    measures the probability that the model will assign a randomly chosen
    positive instance a higher predicted probability compared to a randomly
    chosen negative instance.
    '''
    col1, col2 = st.columns([1,1])
    img = Image.open("ROC-AUC.jpg")
    col1.image(img, width=500)
    col1.caption('The Example Graph of ROC-AUC Classification Evaluation Metric')
    col2.write('**The ROC-AUC Scores**')
    col2.code('''
    # Predict the probabilities for the positive class
    y_pred_logreg = fit_clf_lg.predict_proba(X_test)[:,1]
    y_pred_rf = fit_clf_rf.predict_proba(X_test)[:,1]
    y_pred_dt= fit_clf_dt.predict_proba(X_test)[:,1]
    y_pred_mlp= fit_clf_mlp.predict_proba(X_test)[:,1]

    # Calculate the ROC-AUC scores
    auc_logreg = roc_auc_score(y_test,y_pred_logreg)
    auc_rf = roc_auc_score(y_test,y_pred_rf)
    auc_dt = roc_auc_score(y_test,y_pred_dt)
    auc_mlp = roc_auc_score(y_test,y_pred_mlp)

    print(f"AUC-ROC for Logistic Regression: {auc_logreg}")
    print(f"AUC-ROC for Random Forest: {auc_rf}")
    print(f"AUC-ROC for Decision Tree: {auc_dt}")
    print(f"AUC-ROC for MLP: {auc_mlp}")
    ''')
    col1, col2 = st.columns([1,1])
    col1.write('**The ROC Analysis Graph**')
    col1.code('''
    # Define ROC Analysis
    def plot_roc_curves(y_pred,y_pred_logreg,y_pred_rf,y_pred_dt,y_pred_mlp):
        plt.figure(figsize=(8, 6))

        # Calculate ROC curves for each model
        fpr_logreg, tpr_logreg, _ = roc_curve(y_test,y_pred_logreg)
        fpr_rf, tpr_rf, _ = roc_curve(y_test,y_pred_rf)
        fpr_dt, tpr_dt, _ = roc_curve(y_test,y_pred_dt)
        fpr_mlp, tpr_mlp, _ = roc_curve(y_test,y_pred_mlp)

        # Plot ROC curves
        plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {auc_logreg:.2f})')
        plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
        plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})')
        plt.plot(fpr_mlp, tpr_mlp, label=f'MLP (AUC = {auc_mlp:.2f})')

        # Plot random classifier
        plt.plot([0,1],[0,1], linestyle='--', color='gray')

        # Format the plot
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Heart Disease Prediction Models')
        plt.legend()
        plt.show()

    plot_roc_curves(y_test, y_pred_logreg, y_pred_rf, y_pred_dt, y_pred_mlp)
    ''')
    img = Image.open("ROC-AUC.png")
    col2.image(img, width=500)
    col2.caption('The ROC-AUC Classification Evaluation Metric')
    col2.write('**The ROC-AUC Comparison**')
    comp={'Model':['Logistic Regression', 'Random Forest', 'Decision Tree', 'MLP'],
       'ROC AUC':[0.88,0.90,0.85,0.89]}
    df=pd.DataFrame(comp)
    col2.dataframe(df, hide_index=True, width=500)

    
    '''**Defining The Best Threshold**'''
    col1, col2 = st.columns([1,1])
    col1.code('''
    # ROC Analysis Graph for defining treshold
    def find_rates_for_thresholds(y_test, y_pred, thresholds):
        fpr_list = []
        tpr_list = []
        for threshold in thresholds:
            y_pred_binary = (y_pred > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
            fpr = fp / (fp + tn)
            tpr = tp / (tp + fn)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
        return fpr_list, tpr_list

    thresholds = np.arange(0, 1.1, 0.1)

    fpr_logreg, tpr_logreg = find_rates_for_thresholds(y_test, y_pred_logreg, thresholds)
    fpr_rf, tpr_rf = find_rates_for_thresholds(y_test, y_pred_rf, thresholds)
    fpr_dt, tpr_dt = find_rates_for_thresholds(y_test, y_pred_dt, thresholds)
    fpr_mlp, tpr_mlp = find_rates_for_thresholds(y_test, y_pred_mlp, thresholds)

    # DataFrame Summary
    summary_df = pd.DataFrame({
        'Thresholds':thresholds,
        'FPR_Logreg':fpr_logreg,
        'TPR_Logreg':tpr_logreg,
        'FPR_RF':fpr_rf,
        'TPR_RFg':tpr_rf,
        'FPR_DT':fpr_dt,
        'TPR_DT':tpr_dt,
        'FPR_MLP':fpr_mlp,
        'TPR_MLP':tpr_mlp
    })
    print(summary_df)
    ''')
    col2.code('''
    # ROC Analysis Graph for defining the best treshold
    def find_best_threshold(y_test, y_pred):
        # based on Youden's Index
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]

    best_threshold_logreg = find_best_threshold(y_test, y_pred_logreg)
    best_threshold_rf = find_best_threshold(y_test, y_pred_rf)
    best_threshold_dt = find_best_threshold(y_test, y_pred_dt)
    best_threshold_mlp = find_best_threshold(y_test, y_pred_mlp)

    print(f"Best threshold for Logistic Regression: {best_threshold_logreg}")
    print(f"Best threshold for Random Forest: {best_threshold_rf}")
    print(f"Best threshold for Decision Tree: {best_threshold_dt}")
    print(f"Best threshold for MLP: {best_threshold_mlp}")
    ''')
    col2.write('**The Model Comparison Based on Accuracy, ROC-AUC, Best Threshold**')
    comp={'Model':['Logistic Regression', 'Random Forest', 'Decision Tree', 'MLP'],
       'Accuracy':[0.84,0.86,0.81,0.84],
       'ROC AUC':[0.88,0.90,0.85,0.89],
       'Best Threshold':[0.42,0.64,0.61,0.43]}
    df=pd.DataFrame(comp)
    col2.dataframe(df, hide_index=True, width=500)
    col2.write('''
    It is found that the best accuracy and ROC/AUC analysis are the Random Forest model,
    which is worth 86% and 90% respectively. While the best threshold produced by
    Random Forest is 64%. A high threshold can reduce the error in predicting negative
    cases as positive (False Positive), but can reduce the sensitivity of the model
    (causing more True Negatives to be wrongly predicted as negative).
    ''')    


def heart():
    st.write("""
    Application for the **Heart Disease** prediction
    """)
    st.sidebar.header('User Input Features:')
    # Collects user input features into dataframe
    
    def user_input_features():
        st.sidebar.header('Manual Input')
        cp = st.sidebar.slider('Chest pain type', 1,4,2)
        if cp == 1.0:
            wcp = "Angina-type chest pain"
        elif cp == 2.0:
            wcp = "Unstable pain-type of chest pain"
        elif cp == 3.0:
            wcp = "Severe unstable pain-type chest pain"
        else:
            wcp = "Chest pain not related to heart problems"
        st.sidebar.write("Type of chest pain felt by the patient: ", wcp)
        thalach = st.sidebar.slider("Maximum heart rate achieved", 71, 202, 80)
        slope = st.sidebar.slider("ST-segment slope on electrocardiogram (ECG)", 0, 2, 1)
        oldpeak = st.sidebar.slider("How much ST segment decline or depression", 0.0, 6.2, 1.0)
        exang = st.sidebar.slider("Exercise induced angina", 0, 1, 1)
        ca = st.sidebar.slider("Number of major vessels", 0, 3, 1)
        thal = st.sidebar.slider("Thalium test result", 1, 3, 1)
        sex = st.sidebar.selectbox("Gender", ('Female', 'Male'))
        if sex == "Female":
            sex = 0
        else:
            sex = 1 
        age = st.sidebar.slider("Age", 29, 77, 30)
        data = {'cp': cp,
                'thalach': thalach,
                'slope': slope,
                'oldpeak': oldpeak,
                'exang': exang,
                'ca':ca,
                'thal':thal,
                'sex': sex,
                'age':age}
        features = pd.DataFrame(data, index=[0])
        return features
    
    input_df = user_input_features()
    img = Image.open("heart-disease.jpg")
    st.image(img, width=500)
    if st.sidebar.button('Predict!'):
        df = input_df
        st.write(df)
        with open("generate_heart_disease.pkl", 'rb') as file:  
            loaded_model = pickle.load(file)
        prediction = loaded_model.predict(df)        
        result = ['No Heart Disease' if prediction == 0 else 'Yes Heart Disease']
        st.subheader('Prediction: ')
        output = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(4)
            st.success(f"Prediction of this app is {output}")


def about():
    st.header('About the Creator')
    col1, col2 = st.columns([1,2])
    img = Image.open("profile.png")
    col1.image(img, width=300)
    col2.subheader('**Izzuddin Al Qossam**')
    col2.write('''
    Master of Mechanical Engineering, University of Indonesia
    
    Data science enthusiast
    
    [LinkedIn](https://id.linkedin.com/in/izzuddin-al-qossam-04970589)
               
    [Facebook](https://facebook.com/okazzam)
    
    [Instagram](https://instagram.com/izzuddin.aq20)
    ''')


if add_selectitem == "Overview":
    overview()
elif add_selectitem == "Data Preprocessing":
    dataprep()
elif add_selectitem == "Model Analysis":
    model()
elif add_selectitem == "Prediction":
    heart()
elif add_selectitem == "About":
    about()
