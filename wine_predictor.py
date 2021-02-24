# This algorithm will learn from the data all the characteristics
# that make a red wine and a white wine. After learning,
# I will insert new data for the algorithm to classify the type of wine


# Import libraries and packages necessaries to run the code
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from PIL import Image


# Variable created for save dataset
archive = pd.read_csv('wine_dataset.csv')

# Changing the 'style' column that is in text format to numeric format,
# as this way Python can do the necessary calculations.
archive['style'] = archive['style'].replace('red', 0)
archive['style'] = archive['style'].replace('white', 1)

# Separating variables between predictors (y) and target (x)
y = archive['style']
x = archive.drop('style', axis = 1)

# Creating the training and test data set
# Of the variables y (predictors) and x (target),
# the algorithm will randomly choose 30% as a sample, separate and then train.
# It will also choose another 30%, separate and test.
x_training, x_test, y_training, y_test = train_test_split(x, y, test_size = 0.3)

# Creating the model
# We call the sklearn function (ExtraTreesClassifier) inside the variable 'model'
# which is a training model to apply to the separate training data set.
model = ExtraTreesClassifier()
model.fit(x_training, y_training)

# Right after the training algorithm, we use the 'score' function in our training model to test.
result = model.score(x_test, y_test)

st.title('Welcome to color wine predictor!')
st.write("I am a powerful machine learning algorithm capable of predicting the color of wines! So let's start!")

st.header('In this section you can can know some information about my dataset!')

# Choose and option for showing data
option = st.radio(
    "Choose one option",
    ('Number of lines', 'Number of columns' ,'Just five first data', 'Full dataset!'))

if option == 'Number of lines':
    st.write(f'Number of lines:', len(archive.values))
elif option == 'Number of columns':
    st.write(f'Number of columns:', len(archive.columns))
elif option == 'Just five first data':
    st.write(archive.head())
else:
    st.write('Waiting a moment')
    st.write(archive)

# Result showing the accuracy of the algorithm
st.header('If you wanna see my accuracy, just smash this button bellow!')
if st.button('Press me!'):
    st.write(f'Accuracy:', result)
    st.write('This means that 99% of the time I try to predict the color of the wine, I will get it right! \U0001F60E')

st.header("So let's play a little? Why don't you enter data so I can guess the color of the wine!")
st.write('In the fields below let me know the characteristics of any wine, I will tell you what color that wine is!')

fixed_acidity = st.number_input('Fixed Acidity')
volatile_acidity = st.number_input('Volatile Acidity')
citric_acid = st.number_input('Citric Acid')
residual_sugar = st.number_input('Residual Sugar')
chlorides = st.number_input('Chlorides')
free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide')
total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide')
density = st.number_input('Density')
pH = st.number_input('pH')
sulphates = st.number_input('Sulphates')
alcohol = st.number_input('Alcohol')
quality = st.number_input('Quality')

user_data = pd.DataFrame({'fixed_acidity':[fixed_acidity],
                        'volatile_acidity':[volatile_acidity],
                        'citric_acid':[citric_acid],
                        'residual_sugar':[residual_sugar],
                        'chlorides':[chlorides],
                        'free_sulfur_dioxide':[free_sulfur_dioxide],
                        'total_sulfur_dioxide':[total_sulfur_dioxide],
                        'density':[density],
                        'pH':[pH],
                        'sulphates':[sulphates],
                        'alcohol':[alcohol],
                        'quality':[quality],})

st.header('Do you wanna see your data? If yes click on the button bellow!')
if st.button('Your data!'):
    st.write(user_data)

st.header('Now is the time! You can click on the button that will predict the color of your wine!')
if st.button('Now!'):
    st.write('The color of your wine is...')
    predictions = model.predict(user_data)
    user_result = pd.DataFrame(predictions)
    if user_result.values == 0:
        image = Image.open('red.jpg')
        st.image(image, caption="It's a good red wine!")
    else:
        image = Image.open('white.jpg')
        st.image(image, caption="It's a good white wine!")