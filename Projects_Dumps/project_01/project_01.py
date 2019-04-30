#start with our imports
from scikit-learn import tree

#create our custom dataset which consists of height, weight and shoe size 
#illustrated as a list of lists.

#X partition
X = [[180, 80, 44], [177, 70, 42], [160, 60, 38], [154, 54,37],
        [166, 65, 40], [190, 90, 43], [175, 64, 39], [177, 70, 40],
        [159, 30, 60], [171, 75, 42], [181, 85, 43]]

#Y partition
Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male',
        'female', 'male', 'female', 'male']

#initiate the classifier
clf = tree.DecisionTreeClassifier()

#fit the data into our classifier
clf.fit(X, Y)

#create a prediction
prediction = clf.predict([[190, 70, 43]])
print(prediction)
