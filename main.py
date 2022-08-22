from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from nn_classifier import SimpleNNClassifier


digits = load_digits()
X = digits['data']
y = digits['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=False
)

simple_nn_classifier = SimpleNNClassifier()
random_forest_classifier = RandomForestClassifier(n_estimators=1000)

simple_nn_classifier.fit(X_train, y_train)
random_forest_classifier.fit(X_train, y_train)

predictions = [simple_nn_classifier.predict(X_test), random_forest_classifier.predict(X_test)]

# Доля неправильных ответов
task_1_answer = 1 - accuracy_score(y_test, predictions[0])

# на всякий случай проверю себя, правильно ли я помню суть accuracy
task_1_answer - sum(y_test != predictions[0]) / len(y_test)

task_2_answer = 1 - accuracy_score(y_test, predictions[1])


print(f'Ответ на первое задание: {task_1_answer}')
print(f'Ответ на второе задание: {task_2_answer}')
