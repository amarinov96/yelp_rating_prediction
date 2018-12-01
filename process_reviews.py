import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def process_reviews():

    # read file in batches with pandas
    batch_size = 100000
    total_minibatches = 60
    
    # sets to hold the users and businesses
    users = set()
    businesses = set()
    user_review_pairs = set()

    # list to hold different ratings
    ratings = [0,0,0,0,0]

    # defaultdict to store ratings by year
    ratings_by_year = defaultdict(list)
    ratings_by_month = defaultdict(list)
    
    count=0
    # process dataset in minibatches
    for batch in pd.read_json('yelp_academic_dataset_review.json', lines=True, chunksize=batch_size):

        # progress
        print("Processing minibatch ", count+1, "/", total_minibatches)

        # iterate through current minibatch to collect data
        for index, row in batch.iterrows():

            # get current user and business for review
            curr_user = row['user_id']
            curr_business = row['business_id']
            curr_rating = row['stars']

            # split date into list of year, month, date and add rating to current year ratings
            review_date = str(row['date']).split('-')
            ratings_by_year[review_date[0]].append(curr_rating) 

            # process months < 10 to be single digit
            if review_date[1][0] == '1':
                review_month = review_date[1]
            else:
                review_month = review_date[1][1]

            # append to ratings for this month
            ratings_by_month[review_month].append(curr_rating)

            # add current user and business to sets
            users.add(curr_user)
            businesses.add(curr_business)

            # add to pairs of user - business review
            user_review_pairs.add((curr_user, curr_business))

            # increment number of ratings for these stars
            ratings[curr_rating - 1] += 1

        count+=1

    # Print statistics
    print("\nTotal users: ", len(users))
    print("Total businesses: ", len(businesses))
    print("Total reviews: ", len(user_review_pairs), "\n")
    
    # plots -- uncomment for statistics and plots
    #plot_ratings(ratings)
    #plot_ratings_by_year(ratings_by_year)
    #plot_ratings_by_month(ratings_by_month)

    return users, businesses, user_review_pairs


def plot_ratings_by_month(ratings):

    # y-axis for plot to hold average rating per month
    y = []

    # iterate for all months 1 - 12
    for i in range(1, 13):

        # get count for reviews for current month
        count = len(ratings[str(i)])

        # calculate average rating for this month
        avg_rating = sum(ratings[str(i)]) / count

        # append to list
        y.append(avg_rating)

        # print statistic
        print("Number of reviews for month %d: %d" % (i, count))

    # arrange axes for graph
    x = np.arange(len(y))

    # plot graph and labels
    plt.plot(x, y)
    plt.xticks(x, ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
    plt.ylim(1, 5)
    plt.xlabel('Month')
    plt.ylabel('Average rating')
    plt.title('Average rating per month')
    plt.show()


def plot_ratings_by_year(ratings):

    # y-axis for plot to hold average rating per year
    y = []

    # iterate for all years from 2004 - 2018
    for i in range(2004, 2019):

        # get count of reviews for current year
        count = len(ratings[str(i)])

        # calculate average rating for this year
        avg_rating = sum(ratings[str(i)]) / count

        # append to list
        y.append(avg_rating)

        # print statistic
        print("Number of reviews for %d: %d" % (i, count))

    # arrange axes for graph
    x = np.arange(len(y))

    # plot graph and labels
    plt.plot(x, y)
    plt.xticks(x, list(range(2004,2019)))
    plt.ylim(1, 5)
    plt.xlabel('Year')
    plt.ylabel('Average rating')
    plt.title('Average rating per year')
    plt.show()


def plot_ratings(ratings):

    # print statistics
    for i in range(len(ratings)):
        print("Number of %d star ratings: %d" % (i+1, ratings[i]))

    # arrange axes for bar plot
    x = np.arange(5)

    # plot bar and label
    plt.bar(x, ratings, width=0.35)
    plt.xticks(x, ('1.0', '2.0', '3.0', '4.0', '5.0'))
    plt.xlabel('Ratings')
    plt.ylabel('Count of each rating')
    plt.title('Number of star ratings for all reviews')

    # Display plot of ratings
    plt.show()

# main
if __name__=="__main__":

    users, businesses, user_review_pairs = process_reviews()
    



    

