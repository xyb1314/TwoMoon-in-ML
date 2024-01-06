from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 生成合成数据 (two-moons dataset)
X, y = make_moons(n_samples=2000, noise=0.15, random_state=42)

# 将数据分割成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the MLPClassifier 6 3 2 is best hyperParas for MLP
mlp = MLPClassifier(hidden_layer_sizes=(6,3,2), max_iter=100, alpha=0.0001,
                    solver='adam', random_state=42, activation='relu',
                    learning_rate_init=0.05)

# Train the MLP Classifier
train_accuracies = []
test_accuracies = []
epochs = []

# 绘画分隔平面
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('MLP Classifier Decision Boundary')

for epoch in range(mlp.max_iter):
    mlp.partial_fit(X_train, y_train, classes=np.unique(y))
    train_accuracy = mlp.score(X_train, y_train)
    test_accuracy = mlp.score(X_test, y_test)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    epochs.append(epoch)

# 画准确率变化图 accuracy-epoch relation
plt.figure(1)
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_accuracies, label='Train accuracy')
plt.plot(epochs, test_accuracies, label='Test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epochs')
plt.legend()
plt.show()

plt.figure(2)
plt.figure(figsize=(8, 6))
plot_decision_boundary(mlp, X_test, y_test)
plt.show()