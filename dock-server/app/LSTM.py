# imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import timedelta
import datetime
# allow plots to appear directly in the notebook
import time
import mysql.connector
from mysql.connector import Error

time.sleep(200)
while(1):

    try:
        connection = mysql.connector.connect(host='10.163.234.251',
                                            database='data',
                                            user='root',
                                            password='admin')
        if connection.is_connected():
            db_Info = connection.get_server_info()
            print("Connected to MySQL Server version ", db_Info)
            cursor = connection.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()
            print("You're connected to database: ", record)
            # cursor.execute("INSERT INTO averages VALUES ('"+str(last)+"', "+str(weekly_avg[0][0])+" , 'weeklyAvg') ")
            # cursor.execute("INSERT INTO averages VALUES ('"+str(last)+"', "+str(monthly_avg[0][0])+" , 'monthlyAvg') ")
            #for i in range(len(df)):
            # cursor.execute("INSERT INTO data VALUES ('"+str(df.iloc[0][0])+"', "+df.iloc[0][1])+") ")
            cursor.execute("select * from live223")
            d = cursor.fetchall()
            #print(d)
            #connection.commit()
            #cursor.execute("SELECT * from averages")

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

    import pandas as pd
    live = pd.DataFrame()
    for i in range(len(d)):
        live = live.append(pd.Series([d[i][0],0.01*d[i][1]]), ignore_index = True)

    live.columns = live.columns.astype(str)
    live.rename(columns={"0":"Timestamp","1":"cpuVal"} ,inplace=True)

    df = live.set_index('Timestamp')
    #-----------------------------------------------------------
    # CREATE TRAIN-TEST SPLIT (80:20)
    #-----------------------------------------------------------
    TRAIN_LENGTH = round(len(df)*0.8)
    TEST_LENGTH = len(df) - TRAIN_LENGTH
    train = df.iloc[0:TRAIN_LENGTH]
    test = df[TRAIN_LENGTH : ]
    current_len = len(df)

    #-----------------------------------------------------------
    # GENERATOR TO DATA TO FEED INTO MODEL
    #-----------------------------------------------------------
    def train_generator(dataset, n_lags=1):
        dataX, dataY = [], []
        for i in range(len(dataset)- n_lags -1):
            a = dataset.iloc[i:(i+n_lags)].to_numpy()
            dataX.append(a)
            dataY.append(dataset.iloc[i + n_lags].to_numpy())
        return (np.array(dataX), np.array(dataY))

    #-----------------------------------------------------------
    # DEFINING INPUTS AND EXPECTED OBSERVATIONS

    #-----------------------------------------------------------
    TIME_STEPS = 50
    X_train, y_train = train_generator(train, n_lags = TIME_STEPS)
    X_test, y_test = train_generator(test, n_lags = TIME_STEPS)
    #X_test, y_test = train_generator(test, n_lags = TIME_STEPS)

    import tensorflow as tf
    #-----------------------------------------------------------
    # USING A LSTM MODEL FOR PREDICTION ON TIME SERIES
    #-----------------------------------------------------------
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(512,input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))
    model.add(tf.keras.layers.LSTM(512,return_sequences=False))
    model.add(tf.keras.layers.Dense(1))
    #model.summary()

    model.compile(loss='mean_absolute_error', optimizer= tf.keras.optimizers.Adam())
    #-----------------------------------------------------------
    # DEFINING CALLBACKS
    #-----------------------------------------------------------
    es = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=25, verbose=1, restore_best_weights=True)
    lr_red = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1, min_lr=0.0000001,)

    callbacks = [es,lr_red]
    history = model.fit(X_train, y_train, 
                        epochs=20, 
                        validation_split=0.25,
                        batch_size=128,
                        verbose=1, 
                        shuffle=False,
                        callbacks=callbacks)

    df = df.reset_index()
    last = df.iloc[-1]['Timestamp']

    #PREDICT FOR NEXT 10 DAYS
    import random
    pred_input = np.copy(X_test)
    preds = np.array([1])
    preds = np.delete(preds,0)
    preds_time = {}

    predDf = pd.DataFrame()
    numDays = 10
    intervalsPerDay = 24 * 3
    for i in range(intervalsPerDay*numDays):
        #print(X_test[-1])
        preds = np.append(preds,model.predict(pred_input[len(pred_input)-1:])[0][0])
        preds[-1] = preds[-1] - random.uniform(0, 0.03)
        if preds[-1]>1:
            preds[-1] = 1
        #preds_time[last + pd.Timedelta(minutes = (i+1)*20)] = preds[-1]
        predDf = predDf.append(pd.Series([last + pd.Timedelta(minutes = (i+1)*20),preds[-1]]), ignore_index = True)
        t = np.delete(pred_input[len(pred_input)-1:][0], 0).reshape((49,1))
        pred_input[len(pred_input)-1:][0] = np.append(t, preds[-1]).reshape((50,1))

    predDf.columns = predDf.columns.astype(str)
    predDf.rename(columns={"0":"Timestamp","1":"cpuVal"} ,inplace=True)
    predDf = predDf.set_index('Timestamp')

    predDf = predDf.reset_index()

    try:
        connection = mysql.connector.connect(host='10.163.234.251',
                                            database='data',
                                            user='root',
                                            password='admin')
        if connection.is_connected():
            db_Info = connection.get_server_info()
            print("Connected to MySQL Server version ", db_Info)
            cursor = connection.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()
            print("You're connected to database: ", record)
            # cursor.execute("INSERT INTO averages VALUES ('"+str(last)+"', "+str(weekly_avg[0][0])+" , 'weeklyAvg') ")
            cursor.execute("TRUNCATE TABLE next10;")
            # cursor.execute("INSERT INTO averages VALUES ('"+str(last)+"', "+str(monthly_avg[0][0])+" , 'monthlyAvg') ")
            for i in range(len(predDf)):
            # cursor.execute("INSERT INTO data VALUES ('"+str(df.iloc[0][0])+"', "+df.iloc[0][1])+") ")
                cursor.execute("INSERT INTO next10 VALUES ('"+str(predDf.iloc[i][0])+"', "+str(predDf.iloc[i][1])+") ")
            connection.commit()
            #cursor.execute("SELECT * from averages")

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

    time.sleep(12*60*60)
