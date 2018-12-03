import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.utils import shuffle
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
import re
import math

# set the batch size for processing the users file in batches
batch_size = 200000
minibatches = 8

# set to hold all users
users = set()

# set to hold the users that have more than a certain number of reviews
filtered_users = set()

# the threshold for how many reviews a user must have to be part of the dataset for the model
min_reviews = 300

# process the users file in batches according to batch size
count = 0
print("Processing users dataset...")
for batch in pd.read_json('yelp_academic_dataset_user.json', lines=True, chunksize=batch_size):

    print("Processing minibatch", count+1, " / ", minibatches)

    # shuffle dataset for uniformity
    batch = shuffle(batch)

    # process data in current batch
    for index, row in batch.iterrows():

        # get current user and the total number of reviews he's written
        curr_user = row['user_id']
        num_reviews = row['review_count']

        # add current user to set of all users
        users.add(curr_user)

        # add current user to set of users for the model
        if num_reviews >= min_reviews:
            filtered_users.add(curr_user)

    count+=1

# Statistics for reference
print("Total users: ", len(users))
print("Users with more than ", min_reviews, " reviews: ", len(filtered_users))

# process reviews in batches of 200 000
batch_size = 200000
minibatches = 30

# create dictionary of training examples to convert to pandas dataframe later
fullset_dict = {
        'itemID': [],
        'userID': [],
        'rating': []
        }

print("\nProcessing reviews dataset...")
count=0
# process dataset in minibatches
for batch in pd.read_json('yelp_academic_dataset_review.json', lines=True, chunksize=batch_size):

    # progress
    print("Processing minibatch ", count+1, "/", minibatches)
    
    # shuffle data for uniformity
    batch = shuffle(batch)
    
    # iterate through current batch to process data
    for index, row in batch.iterrows():

        # get current user, business and rating
        curr_user = row['user_id']
        curr_business = row['business_id']
        curr_rating = row['stars']

        # build training lists
        if curr_user in filtered_users:
            fullset_dict['itemID'].append(curr_business)
            fullset_dict['userID'].append(curr_user)
            fullset_dict['rating'].append(curr_rating)

    count += 1

# stats on size of set
print("\nSize of sampled dataset: ", len(fullset_dict['rating']), '\n')

# build pandas df and create reader
# Note: Surprise library requires a Pandas dataframe of a dictionary with keys for user, item(business) and rating 
# as well as a Reader object which indicates the scale of the ratings in order to create a Surprise dataset
reader = Reader(rating_scale=(1,5))
full_df = pd.DataFrame(fullset_dict)

# load dataset from pandas df into Surprise dataset and build trainset
# Note the Surprise dataset takes two arguments: a Pandas dataframe with the specified keys
# (the name of the keys are arbitrary as long as they are consistent with the dictionary names above)
# and a Reader object (from the Surprise library) that specifies the range of the ratings
data = Dataset.load_from_df(full_df[['userID', 'itemID', 'rating']], reader)

# initialize a random seed so that train test split is the same across runs
seed = np.random.RandomState(42)

# split the data into 80% training 20% validation
# Note we pass in the random seed as the second argument to ensure that the 'randomness' of the split
# is consistent every time the file is ran
trainset, testset = train_test_split(data, test_size=.2, random_state=seed)

# Initialize the algorithm that we are going to use to train on the dataset
# Here we use standard SVD algorithm (matrix factorization with user and item biases)
# n_factors specifies the number of factors to be used, n_epochs specifies the number of iterations
# of stochastic gradient descent, and verbose=True gives us progress on the epochs
# Check Surprise documentation on SVD for full list of specifiable parameters
print("Training model...")
algo = SVD(n_factors=50, n_epochs=10, verbose=True)

# This call to fit() on the trainset actually performs the training of the model
algo.fit(trainset)

# The call to test() on the testset makes predictions on ratings of user-items in the testset
# according to the trained model above
predictions = algo.test(testset)

# This line gives us the accuracy in terms of RMSE of the predictions made above
accuracy.rmse(predictions)

# Test again with different params
# Note when we don't specify the number of epochs, the default is 20
print("Training model...")
algo = SVD(n_factors=50, verbose=True)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# Test again with different params
print("Training model...")
algo = SVD(n_factors=50, n_epochs=40, verbose=True)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# Test again with different params
print("Training model...")
algo = SVD(n_factors=10, n_epochs=10, verbose=True)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# Test again with different params
print("Training model...")
algo = SVD(n_factors=10, verbose=True)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# Test again with different params
print("Training model...")
algo = SVD(n_factors=10, n_epochs=40, verbose=True)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# Test again with different params
print("Training model...")
algo = SVD(n_factors=5, n_epochs=10, verbose=True)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# Test again with different params
print("Training model...")
algo = SVD(n_factors=5, verbose=True)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# Test again with different params
print("Training model...")
algo = SVD(n_factors=5, n_epochs=40, verbose=True)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# Run K-fold cross-validation with k=5 to get the final RMSE 
# K-fold cross-validation gives us the final unbiased RMSE score for our model
# as it runs training and validation on 5 different training/validation splits of the original data
# and takes the average
# Note we run this after we've established the optimal parameters for the model from earlier runs
# for the final result - in this case n_factors=5, n_epochs=20
algo = SVD(n_factors=5, verbose=True)
cross_validate(algo, data, measures='RMSE', cv=5, verbose=True)





