#organize our imports
import numpy as np
from lightfm.datasets import fetch_movielens

#load our dataset
data = fetch_movielens(min_rating=4.0)

"""
print the distribution percentages between the train split and the
test split
print(repr(data['train']))
print(repr([data['test']]))
"""

#initialize our model
model = LightFM(loss='warp')

#start the training project
model.fit(data['train'], epochs=30, num_thread=2)

def simple_recommendation(model, data, user_ids):
    n_user, n_items = data['train'].shape
    for user_id in user_ids:
        known_positive = data['item_labels'][data['train'].tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_label'][np.argsort(-scores)]
        print("User %s" % user_id)
        print("Known Positives:")

        for x in known_positives[:3]:
            print(" %s " % x)
        print("Recommended Choices")

        for x in top_items[:3]:
            print(" %s " % x)

sample_recommendation(model, data, [3, 25, 450])
