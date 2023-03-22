
![Screenshot (454)](https://user-images.githubusercontent.com/121576163/226862172-63ad1f42-1c1d-4058-9774-dc0d747d0e02.png)

# Stock Sentiment Analysis Using News Headlines

In this project, I will try to predict whether the stock price increases or decreases based on top 25 news headlines of the day.

## About the Project

It is widely accepted that media, news and publicity can have a profound effect on stock prices. To check this hypothesis, I decided to carry out this project.

Dow Jones Industrial Average is a stock market index that collects the value of a list of 30 large and public companies based in the US. In this way it gives an idea of the trend that is going through the stock market.

News and global events, political or otherwise, play a major role in changing stock values. Every stock exchange is, after all, reflects how much trust investors are ready to put in other companies.

## Data Overview

Source: [Kaggle](https://www.kaggle.com/datasets/aaron7sun/stocknews)

Data: This data contains 8 years of daily news headlines from 2000 to 2016 from Yahoo Finance and Reddit WorldNews, as well as the Dow Jones Industrial Average(DJIA) close value of the same dates as the news.

This is a binary classification problem. When the target is “0”, the same day DJIA close value decreased compared with the previous day, when the target is “1”, the value rose or stayed the same.

Downloaded the csv file from Kaggle.

## Data Attributes

1. Date: Date column contains the dates from 2000 to 2016 on which the news are released.

2. Label: Binary Numeric, '0' represent that the price went down and '1' represent that the price went up.

3. Top#: Strings that contains the top 25 news headlines for the day ranging from 'Top1' to 'Top25'

## Data Snapshot

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

## Train Test Split

Splitted the data at the very start so that there is no scope of information seeping from train dataset to test dataset.

```javascript
train1=df[df.Date<'20150101']
test1=df[df.Date>'20141231']
```

## Separating Target and removing date column
```javascript
train=train1.iloc[:,2:27]
test=test1.iloc[:,2:27]
train_label=train1.Label
test_label=test1.Label
```
## Data Cleaning & Preparation

Performed the following steps in order to clean and prepare the data:

1. Removed punctuation marks
2. Converted headlines to lowercase
3. Combined various headlines per day in one sentence
```javascript
train.replace(to_replace="[^a-zA-Z]",value=' ',regex=True,inplace=True)
test.replace(to_replace="[^a-zA-Z]",value=' ',regex=True,inplace=True)
for i in range(0,25):
    train[i]=train[i].str.lower()
    test[i]=test[i].str.lower()
headlines = []
for row in range(0,len(train.index)):
    headlines.append(' '.join(str(x) for x in train.iloc[row,0:25]))
```
## Lemmatization and stopwords removal
```javascript
for i in range(0,len(headlines)):
    words=nltk.word_tokenize(headlines[i])
    words=[lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    headlines[i]=' '.join(words)
```
## Embedding (Bag of Words and TfIdf)

Used Bag of words and TfIdf to embedd headlines as it could affect model performance.

```javascript
countvector=CountVectorizer(analyzer='word')
traindataset=countvector.fit_transform(headlines)

cv=TfidfVectorizer(analyzer='word')
traindataset1=cv.fit_transform(headlines)
```

## Modeling Baseline

First did a further split of train data to train and validation data. Then chose following models for evaluation.

```javascript
lr_model = LogisticRegression(n_jobs=-1)
nb_model = naive_bayes.MultinomialNB()
svc_model = svm.SVC(probability=True, gamma="scale",)
rf_model = ensemble.RandomForestClassifier(n_estimators=100,n_jobs=-1)

models = ["lr_model", "nb_model", "svc_model", "rf_model"]
```

Used the following function to see classification report of the models to select baseline model.

```javascript
def baseline_model_filter(modellist, X, y):
    ''' 1. split the train data further into train and validation (17%). 
        2. fit the train data into each model of the model list
        3. get the classification report based on the model performance on validation data
    '''
    X_train, X_valid, y_train, y_valid = X[:3471],X[3471:],y[:3471],y[3471:]
    for model_name in modellist:
        curr_model = eval(model_name)
        curr_model.fit(X_train, y_train) 
        print(f'{model_name} \n report:{classification_report(y_valid, curr_model.predict(X_valid))}')

models = ["lr_model", "nb_model", "svc_model", "rf_model"]
```
Each model generated a report for both bag of words and tfidf embeddings.

Proceeded with random forest and bag of words embedding. Its report is as follows.
![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

## Hyper-parameter tuning

Used GridsearchCV to tune the hyperparameters.

```javascript
model=RandomForestClassifier(n_jobs=-1,warm_start=True)
param={'n_estimators':[100,200,300],
    'criterion':['gini','entropy']}

gcv=GridSearchCV(model,param_grid=param,n_jobs=-1,scoring='accuracy')
```
Thus obtained the final classification_report.

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Further improvements:

1. The current analysis is based on 16 years of data from 2000 to 2016. I would like to collect more data and recent data to improve the model.
2. News from single relavent source will prove more efficient in building the model. It will ease cleaning and produce more quality data for modelling.
3. In the future I would like to try deep learning model because deep learning works very well in natural language processing problems.












