import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM


data  = fetch_movielens(min_rating=4.0)

print(repr(data["train"]))
print(repr(data["test"]))

# creating a model
model = LightFM(loss="warp")

#  training the model
model.fit(data["train"], epochs=30,  num_threads=2)

def recommendations(model,data,user_ids):

    # num of users and movies in the matrix 
    number_users, number_items = data["train"].shape

    for user_id in user_ids:

        # movies they already like
        liked_movies  = data["item_labels"][data["train"].tocsr()[user_id].indices]

        # movies we predict they will like
        M_list = model.predict(user_id,np.arange(number_items))

        # rank them in  order of most liked to least
        top_items = data["item_labels"][np.argsort(-M_list)]

        print("User %s"  % user_id)
        print("         Liked movies")

        # printing top 3 liked movies
        for x in liked_movies[:3]:
            print("         %s" % x)

        print("\n         Recommended")
        for y in top_items[:3]:
            print("         %s" % y)

recommendations(model,data,[2,4,54])