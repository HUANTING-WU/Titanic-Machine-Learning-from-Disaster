def plot_distribution(data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    plt.figure()
    sns.distplot(data, fit=stats.norm);
    mean, std = stats.norm.fit(data)
    print('mean = {:.4f}\nstd = {:.4f}\nskewness = {:.4f}\nkurtosis = {:.4f}'
          .format(mean, std, data.skew(), data.kurtosis()))
    plt.figure()
    stats.probplot(data, plot=plt);

def describe_numerical(data):
    print('mean:', data.mean())
    print('median:', data.median())
    print('mode:', data.mode().values[0])
    print(data.describe())

def missing_value(data):
    print('missing value number:', data.isnull().sum())
    print('missing value percentage:', data.isnull().sum()/len(data))

def submit(test_X_og, pred_y):
    import pandas as pd

    submit = pd.DataFrame(data=[test_X_og.index, pred_y]).T
    submit.columns = ['PassengerId', 'Survived']
    submit = submit.astype('int32')
    submit.to_csv('submit.csv', index=False)

def gridsearchcv(model, param_grid, train_X, train_Y, dev_X, dev_Y):
    from sklearn.model_selection import (cross_val_score, GridSearchCV, KFold)
    from sklearn.metrics import accuracy_score

    model = GridSearchCV(model, param_grid=param_grid, scoring='accuracy',
                        cv=KFold(n_splits=5, shuffle=True, random_state=None))
    model.fit(train_X, train_Y)
    print('grid search best parameters:', model.best_params_)
    print('grid search best scores: {:.4f}'.format(model.best_score_))

    train_scores = cross_val_score(model, train_X, train_Y, scoring='accuracy',
                                cv=KFold(n_splits=5, shuffle=True, random_state=None))
    train_score = train_scores.mean()
    print('cv score: {:.4f}'.format(train_score))

    pred_y = model.predict(dev_X)
    dev_score = accuracy_score(dev_Y, pred_y)
    print('dev score: {:.4f}'.format(dev_score))

    return model

def plot_result(history):
    import matplotlib.pyplot as plt

    print('train set loss: {:.4f}'.format(history.history['loss'][-1]))
    print('dev set loss: {:.4f}'.format(history.history['val_loss'][-1]))
    print('train set accuracy: {:.4f}'.format(history.history['binary_accuracy'][-1]))
    print('dev set accuracy: {:.4f}'.format(history.history['val_binary_accuracy'][-1]))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Dev Loss'], loc='upper right')
    plt.show()

    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Loss Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train Accuracy', 'Dev Accuracy'], loc='upper right')
    plt.show()

def create_nn(train_X, train_Y, dev_X, dev_Y, l1, l2, lr, batch_size, epochs):
    from keras.callbacks import (History, EarlyStopping)
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras import losses
    from keras import metrics
    from keras import optimizers
    from keras import initializers
    from keras import regularizers

    model = Sequential()

    model.add(Dense(64, activation='relu',
              kernel_initializer=initializers.he_normal(seed=42),
              bias_initializer=initializers.Zeros(),
              kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
              bias_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))

    model.add(Dropout(rate=0.5, seed=42))

    model.add(Dense(32, activation='relu',
              kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
              bias_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.Adam(lr=lr),
              loss=losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])

    history = History()
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(train_X, train_Y, validation_data=[dev_X, dev_Y], shuffle=True, verbose=0,
                batch_size=batch_size, epochs=epochs, callbacks=[history, early_stop])

    return model, history, early_stop
