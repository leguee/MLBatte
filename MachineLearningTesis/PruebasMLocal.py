import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error



# Import the adult.txt file into Python
data = pd.read_csv('data/Prueba1.csv', sep=',')
print (data.corr()) ## TODO analizar que me tira estos valores.
# DO NOT WORRY ABOUT THE FOLLOWING 2 LINES OF CODE
# Convert the string labels to numeric labels
#for label in ['race', 'occupation']:
#    data[label] = LabelEncoder().fit_transform(data[label])

# Take the fields of interest and plug them into variable X
X = data[['day_of_week', 'hour', 'bluetooth','data','min','wifi']]
# Make sure to provide the corresponding truth value
Y = data['battery'].values.tolist()
print ("esto corresponde al X.describe")
print (X.describe())
# Split the data into test and training (30% for test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Instantiate the classifier
clf = LogisticRegression()

# Train the classifier using the train data
clf = clf.fit(X_train, Y_train)

print ('Prediciendo un ejemplo de HOY')
prediction = clf.predict([[0, 23, 1,1,18,0]])

print(prediction)

# Validate the classifier, lo que dice es cuantos se clasificaron como verdaderos positivos y cuantos como verdaderos negativos, dividido el total
# lo que indica es que porcentaje de la data es casificada correctamente -- ver si aplica a regression.
accuracy = clf.score(X_test, Y_test)
print ('Accuracy(exactitud): ' + str(accuracy))

# Make a confusion matrix
prediction = clf.predict(X_test)

cm = confusion_matrix(prediction, Y_test)
print ("imprimiendo confusion matrix")
print (cm)

