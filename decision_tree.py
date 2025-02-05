from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.2f}")

# Display tree structure
print("Decision Tree Structure:")
print(export_text(dt, feature_names=iris.feature_names))

# Visualize the decision tree
plt.figure(figsize=(12,8))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Visualizing feature importance
feature_importances = dt.feature_importances_
plt.figure(figsize=(8,6))
sns.barplot(x=feature_importances, y=iris.feature_names)
plt.xlabel("Feature Importance")
plt.ylabel("Feature Names")
plt.title("Feature Importance in Decision Tree")
plt.show()

# Visualizing dataset distribution
plt.figure(figsize=(8,6))
sns.pairplot(sns.load_dataset("iris"), hue="species")
plt.title("Iris Dataset Distribution")
plt.show()
