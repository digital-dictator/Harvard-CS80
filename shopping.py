import csv
import sys
from pandas import DataFrame, read_csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Read and parse data
    # Make list of months to iterate and replace with index
    Month = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Make Pandas DataFrame with all rows/columns
    df = pd.read_csv("shopping.csv", index_col=0,
                 dtype = {'Administrative': int, 'Administrative_Duration': float, 'Administrative': int, 'Administrative_Duration': float,
                 'Administrative': int, 'Administrative_Duration': float, 'Informational': int, 'Informational_Duration': float,
                 'ProductRelated': int, 'ProductRelated_Duration': float, 'BounceRates': float, 'ExitRates': float,
                 'PageValues': float, 'SpecialDay': float, 'Month': str, 'OperatingSystems': int,
                 'Browser': int, 'Region': int, 'TrafficType': int, 'VisitorType': str, 'Weekend': str, 'Revenue': str})
    # Replace month with index of month
    for row in Month:
        df = df.replace(row, Month.index(row))
    # Replace all data points requested with their respective integers
    df = df.replace("Returning_Visitor", 1)
    df = df.replace("New_Visitor", 0)
    df = df.replace("Other", 0)
    df = df.replace("TRUE", 1)
    df = df.replace("FALSE", 0)
    df = df.replace(True, 1)
    df = df.replace(False, 0)
    # data_tuple = list of evidence and list of Labels i.e. revenue (last column)
    data_tuple = (df.iloc[:,0:16].values.tolist(), df.iloc[:,16].values.tolist())
    return data_tuple


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # model = KNeighborsClassifier(n_neighbors=1)
    model = KNeighborsClassifier(n_neighbors=1)
    # Fit model
    return model.fit(evidence, labels)

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # All positive labels
    positive = 0
    # All negative labels
    negative = 0
    # All positive labels matching with predictions
    true_positive_rate = 0
    # All negative labels matching with predictions
    true_negative_rate = 0
    # Add to each variable
    for i in range(len(labels)):
        if labels[i]:
            positive += 1
        else:
            negative +=1

        if labels[i] == 1 and predictions[i] == 1:
            true_positive_rate += 1
        elif labels[i] == 0 and predictions[i] == 0:
            true_negative_rate += 1
    # Matching predited positive labels over total positive
    sensitivity = true_positive_rate/positive
    # Matching predited negative labels over total negative
    specificty = true_negative_rate/negative

    return (sensitivity, specificty)


if __name__ == "__main__":
    main()
