# Project 3: Web APIs & Classification

### Problem Statement

The goal of this project is twofold 1) To collect post title data from the rock and rap music subreddit categories using Reddit's API and 2) Use the title data in conjunction with a Naive Bayes Classifier and Logistic Regression to train a model that will classify Reddit posts into their respective categories. The two models will be developed using the SciKit-Learn Logistic Regression and Naive Bayes algorithms with hyperparameters and specific forms of Naive Bayes being evaluated and explored based on original model performance. Success will be evaluated by the models ability to accurately classify subreddit threads based on the accuracy score of the testing data. Being able to accurately predict the origin of a subreddit post may be of use to business leaders in the music industry seeking to analyze data trends and cultures of those who post in the rock and rap subreddits for the purpose of influencing music creation, marketing and predicting cultural trends as well as the individual music consumers that stand to benefit from their influence.

#### Python Reddit API Wrapper (PRAW) Scraping

The Python Reddit API Wrapper (PRAW) was used in conjunction with my Reddit credentials in order to pull data through this wrapper and into Pandas dataframes. My reddit credentials were saved as a .json file and an instance of the Reddit class was instantiated using these credentials. A custom function called get_batch was written by taking the name of the desired subreddit and passing it into the function as a string. At this point, the data was pulled 1000 posts at a time and a dictionary containing each constituent part of each subreddit thread post was populated via list comprehension and converted into a Pandas DataFrame to be fed into the EDA and cleaning phase.

#### EDA and Data Cleaning

The original data was first read into Pandas DataFrames. Following, a custom function was defined that used a combination of regular expressions and the text processing library Beautiful soup to remove any unwanted HTML tags, punctuation and stopwords if present. The end result of applying this function was a Pandas DataFrame similar to the one we started with, only cleaing had been applied to the title column. The rest of the information such as comments, age, and thread name remained the same. The data was then exported to separate .csv files to be used in the preprocessing step. I noticed some null values appear in the modeling phase due to subreddit post titles being strings of exclusively punctuation. These rows were dropped.

#### Preprocessing and Visualizations

The cleaned data was read into Pandas DataFrames and null values dropped. A custom function was defined that applied CountVectorizer to the column passed into the function, created a DataFrame of words in the column, found the 10 most common words in that set of data and created a Seaborn bar chart visualizing the most common words for each subreddit. The two DataFrames containing rap and rock data were then combined into one using Pandas concatenate. The output column 'thread' was identified during this step and values 1 and 0 were mapped to 'rock' and 'rap', respectively because this is a binary classification problem intended to predict the class of a subreddit post. The data was then output to a single training dataset to be used for modeling.

#### Modeling

The data was read into a Pandas DataFrame. Classification model variables were then created: X) the model input data, subreddit post titles to be classified. y) the model output data, subreddit title; the 'rock' and 'rap' subreddit name strings were mapped to 1 and 0 respectively in the preprocessing step. Some pesky null values appeared here but the rows containing them were dropped. The data was split into training and testing sets and pipelines were established for gridsearching over (4) different combinations of text vectorizers and binary classification algorithms. The text data had to be converted into a binary representation for use with any binary classifier. The Logistic Regression (LR) model was formed as a baseline for model comparison due to its simplicity and ease of interpretation. The Naive Bayes algorithm was then implemented for its ability to classify text data and evaluated for performance. The results were evaluated by two different criteria 1) the amount of model overfitting as measured by the difference between the training and testing scores and 2) the testing accuracy score. The model chosen was the Bernoulli Naive Bayes model; the type of vectorization had little effect on the model. Summary statistics are shown below:

CVEC/BNB Best Accuracy Score: 0.834
CVEC/BNB Training Score: 0.898
CVEC/BNB Testing Score 0.843

#### Executive Summary

Statistically speaking, a good classification model is going to optimize the bias-variance tradeoff to some degree as well as preserve the ability to accurately predict the classes of new data. Keeping this in mind, the pipeline containing a Bernoulli Naive Bayes model with a simple CountVectorizer was selected as the final model for predicting if a subreddit post originated from the 'rock' or 'rap' subreddits. GridSearching was applied to all of the considered models with the maximum features beaing varied from 100 to 1000, the n-gram (tuples containing different combinations of specific words) range was varied over 1-word, 2-word and 3-word tuples and english stopwords were left in and removed from the model. Gridsearching over higher numbers of features and higher n-gram ranges had little to no effect on the model scores.

After analyzing the most common words contained in both musical genres, some interesting observations were made:

Within the top ten words in the rock genre 'live', 'band' and 'album' stuck out to me because of their implication for how music is created. These words seem to emphasize the live performance aspect of rock music as well as the tendency for a rock band to play full length albums while performing on stage.

Within the rap genre the words 'beat' and 'song' stood out to me for the same reason. I'd like to point out the abbreviated word "prod" short for 'produced by' as an important word in the rap genre used to differentiate between the artist that performed or recorded the song and the artist that produced the song. 

The ability to gather consumer data from a public internet forum, predict the origin of that data and interpret words referring to how that music was created, where that music is performed, the structure of that musical genres typical performance and how people wearing different hats in the music industry exchange ideas could be absolutely telling of new technical trends in music creation. I believe the next best step would be to routinely implement machine learning algorithms in this context in order to continuously gather new text data and use it to predict shifts in the music industry and stay ahead of the curve when it comes to the technology and artistic themes that are used to create it. 



