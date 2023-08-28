import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read the CSV file into a DataFrame
dfCredit = pd.read_csv('creditcard.csv')

# Display the first few rows of the DataFrame
# print(dfCredit.head())

# Check the data types of columns
# print(dfCredit.dtypes)

# Check for missing values in the DataFrame
# print(dfCredit.isnull().sum())

# Separate non-fraud and fraud transactions
# dfNotFraud = dfCredit.Amount[df_credit.Class == 0]
# dfIsFraud = dfCredit.Amount[df_credit.Class == 1]

# Check the distribution of fraud and non-fraud transactions
# print(dfCredit.Class.value_counts())

# Analyze descriptive statistics of non-fraud transactions
# print(dfNotFraud.describe())

# Analyze descriptive statistics of fraud transactions
# print(dfIsFraud.describe())

# Separate non-fraud and fraud transactions into two DataFrames
dfNotFraud = dfCredit[dfCredit.Class == 0]
dfIsFraud = dfCredit[dfCredit.Class == 1]

# Sample a subset of non-fraud transactions (492 samples)
dfSampleNotFraud = dfNotFraud.sample(492)

# Concatenate the sampled non-fraud transactions with fraud transactions
df = pd.concat([dfSampleNotFraud, dfIsFraud], axis=0)

# Reset the index of the concatenated DataFrame
df.reset_index(inplace=True)

# Take a subset of the first 5 rows for validation of non-fraud transactions
dfValNotFraud = df.head(5)

# Take a subset of the last 5 rows for validation of fraud transactions
dfValIsFraud = df.tail(5)

# Remove the first 5 rows from the main DataFrame
df = df.iloc[5:]

# Remove the last 5 rows from the main DataFrame
df = df.iloc[:-5]

# Concatenate the validation DataFrames for non-fraud and fraud transactions
dfValTotal = pd.concat([dfValNotFraud, dfValIsFraud])

# Reset the index of the concatenated validation DataFrame
dfValTotal.reset_index(inplace=True)

# Extract the 'Class' column from the validation DataFrame for later comparison
dfValTotalReal = dfValTotal.Class

# Drop unnecessary columns from the validation DataFrame
dfValTotal = dfValTotal.drop(['level_0', 'index', 'Time', 'Class'], axis=1)

# Verify the distribution of fraud and non-fraud transactions in the main DataFrame
# print(df.Class.value_counts())

# Prepare the features (x) and target (y) for the machine learning model
x = df.drop(['index', 'Time', 'Class'], axis=1)
y = df['Class']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train a logistic regression model
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)

# Make predictions on the test set
pred = lr.predict(x_test)

# Calculate accuracy of the model
acc = accuracy_score(y_test, pred)

# Create a result string for printing
resultString = f'Accuracy: {acc:.2f}'
# print(resultString)

# Perform predictions on the validation data
pred = lr.predict(dfValTotal)

# Create a DataFrame for comparing real values and predictions
df = pd.DataFrame({'real': dfValTotalReal, 'prediction': pred})

# Print the DataFrame
print(df)
