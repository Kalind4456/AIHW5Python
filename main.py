import numpy as np
import matplotlib.pyplot as plt
import argparse


def parse_example(line):
    """ Parse each line of the dataset and return the label and the feature dictionary. """
    parts = line.split()
    label = int(parts[0])
    features = {}
    for item in parts[1:]:
        index, value = item.split(':')
        features[int(index)] = int(value)
    return label, features


def dot_product(weights, features):
    """ Compute the dot product manually for sparse representation. """
    return sum(weights.get(index, 0) * value for index, value in features.items())


def sigmoid(z):
    """ Compute the sigmoid function. """
    return 1 / (1 + np.exp(-z))


def update_weights(weights, features, label, alpha):
    """ Update the weights using stochastic gradient descent (SGD). """
    prediction = sigmoid(label * dot_product(weights, features))
    for index, value in features.items():
        weights[index] = weights.get(index, 0) + alpha * (1 - prediction) * label * value


def train_model(data, alpha, epochs):
    """ Train logistic regression model using SGD. """
    weights = {}
    for epoch in range(epochs):
        np.random.shuffle(data)  # Shuffle data for each epoch
        for line in data:
            label, features = parse_example(line)
            update_weights(weights, features, label, alpha)
    return weights


def predict(weights, features):
    """ Predict the class label for given features. """
    z = dot_product(weights, features)
    return 1 if sigmoid(z) >= 0.5 else -1


def evaluate_model(data, weights):
    """ Evaluate the model on the data and return accuracy. """
    correct = 0
    total = len(data)
    for line in data:
        label, features = parse_example(line)
        prediction = predict(weights, features)
        if prediction == label:
            correct += 1
    return correct / total


def plot_learning_curve(training_data, testing_data, alpha, epochs):
    """ Plot the learning curve of updates vs accuracy on test data. """
    weights = {}
    accuracies = []
    for epoch in range(epochs):
        np.random.shuffle(training_data)  # Shuffle data for each epoch
        for line in training_data:
            label, features = parse_example(line)
            update_weights(weights, features, label, alpha)
        accuracies.append(evaluate_model(testing_data, weights))
    plt.plot(range(epochs), accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.show()


def load_data(file_name):
    """ Load data from a file. """
    with open(file_name, 'r') as file:
        data = file.readlines()
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate logistic regression model.")
    parser.add_argument("--train", type=str, required=True, help="File path for training data.")
    parser.add_argument("--dev", type=str, required=True, help="File path for development data.")
    parser.add_argument("--test", type=str, required=True, help="File path for testing data.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")

    args = parser.parse_args()

    training_data = load_data(args.train)
    dev_data = load_data(args.dev)
    test_data = load_data(args.test)

    weights = train_model(training_data, args.alpha, args.epochs)
    print("Development set accuracy:", evaluate_model(dev_data, weights))
    print("Test set accuracy:", evaluate_model(test_data, weights))

    plot_learning_curve(training_data, test_data, args.alpha, args.epochs)
