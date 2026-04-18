from google.cloud import bigquery 
from sklearn.model_selection import train_test_split

client = bigquery.Client()
def fetch_data(query):
    result = client.query(query,).to_arrow()
    df=result.to_pandas()
    return df

def split_data(df, test_size=0.2,val_size=0.5, random_state=42):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=y)
    X_val,X_test,y_val,y_test = train_test_split(X_test,
                                                y_test,
                                                test_size=val_size,
                                                random_state=random_state,
                                                stratify=y_test)
    return X_train,X_val, X_test, y_train,y_val, y_test

def scale_features(X_train,X_val, X_test,scaler):
    """Scale Amount et Time avec RobustScaler"""
    
    cols_to_scale = ['Amount', 'Time']
    
    # fit UNIQUEMENT sur train — jamais sur test
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    
    # transform seulement sur test et val
    X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    return X_train,X_val, X_test, scaler