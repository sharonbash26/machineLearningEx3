import numpy as np
import random

"""
v is vector
"""

# relu is a function in a vector where has koridinta smaller than 0 she puts 0.
def ReLU(v):
    return (np.abs(v) + v) / 2


def ReLU_der(v):
    return np.greater(v, 0).astype(int)

#this function does nirmullation to the last vector
def softmax(v):
    c = np.amax(v)
    exp = np.exp(v - c)
    return exp / np.sum(exp)  # vector hlak number


def compute_loss_and_gradients(model, x, y):  # compute loss and nigzrat of one example
    W1, b1, W2, b2 = model  # open model
    h = ReLU(np.dot(x, W1) + b1)
    output = np.dot(h, W2) + b2
    probs = softmax(output)

    loss = -np.log(probs[y])  # the protabliity of the right class

    softmax_der = np.dot(W2, probs) - W2[:, y]  # nigzrat of soft max
    gW2 = np.outer(h, probs)  # girdiant of w2
    gW2[:, y] -= h

    gb2 = probs
    gb2[y] -= 1

    relu_der = ReLU_der(h)
    gW1 = np.outer(x, relu_der) * softmax_der
    gb1 = softmax_der * relu_der

    return loss, [gW1, gb1, gW2, gb2]  # this is function compute the giradiant of w1 w2 b1 b2


def train_epoch(model, X, Y, learning_rate):
    indices = list(range(len(X)))
    random.shuffle(indices)   # balgan of example

    total_loss = 0
    for i in indices: # run on amount of phots
        x, y = X[i], Y[i] # take spacpic example
        loss, gradients = compute_loss_and_gradients(model, x, y)
        total_loss += loss

        for model_index in range(len(model)): # update mishkolt
            model[model_index] = -learning_rate * gradients[model_index]
        # the code above updates the model with the gradients
        # W1 -= learning_rate * gW1
        # b1 -= learning_rate * gb1
        # W2 -= learning_rate * gW2
        # b2 -= learning_rate * gb2
    return total_loss / len(indices) # return the average loss

def forward(model, x):
    W1, b1, W2, b2 = model  # open model
    h1 = ReLU(np.dot(x, W1) + b1)  # x vector of picsl
    output = np.dot(h1, W2) + b2
    return softmax(output)


def predict(model, x):
    return np.argmax(forward(model, x))


def compute_accuracy(model, X, Y):  # compute how mush model say true
    good = bad = 0
    for (x, y) in zip(X,Y):
        y_hat = predict(model, x)
        if y == y_hat:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def init_matrix(m, n):  # m n size of matrix
    shape = (m, n)
    if m == 1 or n == 1:
        if m == 1:
            shape = (n,)  # vector in size n
        else:
            shape = (m,)

    xavier_init_value = np.sqrt(6 / (m + n))
    return np.random.uniform(-xavier_init_value, xavier_init_value, shape)


def init_model(in_dim, hid_dim, out_dim):
    W1 = init_matrix(in_dim, hid_dim)
    b1 = init_matrix(1, hid_dim)

    W2 = init_matrix(hid_dim, out_dim)
    b2 = init_matrix(1, out_dim)

    return [W1, b1, W2, b2]


def read_x_file(filename):
    x = []
    with open(filename, 'r') as f:
        for line in f:
            splitted = line.strip().split()  # strip delete /n . split mpcel to vuale
            vector = np.array(splitted, dtype=np.int) / 256  # nirmol
            x.append(vector)
    return x


def read_y_file(filename):
    with open(filename, 'r') as f:
        y = [int(line.strip()) for line in f]
    return y


def start():
    train_x = read_x_file('train_x')
    train_y = read_y_file('train_y')

    split = int((len(train_x) * 8) / 10)
    dev_x = train_x[:split]
    dev_y = train_y[:split]

    train_x = train_x[split:]
    train_y = train_y[split:]

    model = init_model(784, 100, 10)

    for epoch in range(25):
        loss = train_epoch(model, train_x, train_y, 0.03)
        train_acc = compute_accuracy(model, train_x, train_y) # how much we succed in train
        dev_acc = compute_accuracy(model, dev_x, dev_y) # how ouch we succed in validation
        print('Epoch: {0}, train: {1}%, dev: {2}%, loss: {3:.2f}'
              .format(epoch+1,int(100*train_acc), int(100*dev_acc), loss))


    test_x = read_x_file('test_x')
    test_y = [predict(model, x) for x in test_x]
    with open('test_y', 'w') as f:
        f.write('\n'.join(str(y) for y in test_y))


    print()


if __name__ == '__main__':
    start()
