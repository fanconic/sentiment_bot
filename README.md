# sentiment_bot
Telegram ChatBot which replies if a sent message is a good or bad movie review.

The ChatBot has been trained on the IMDb movie review DataSet which can be found here: http://ai.stanford.edu/~amaas/data/sentiment/.

There are different machine learning models used. Mainly it's various combinations of Naive Bayes and Logistic Regression with different numbers of BiGrams, as well as vectorizers.

The best accuracy was achieved by logistic regression with a bigram of range 2, and a normal CountVectorizer.
The libraries used is mainly sklearn.
