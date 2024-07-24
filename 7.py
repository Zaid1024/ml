import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

file_path = r"dataset/weather_forecast.csv"
df = pd.read_csv(file_path)

print(df.head())

encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(df.drop('Play', axis=1)).toarray()
y = df['Play']

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.24, random_state=42)

clf_id3 = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_id3.fit(X_train, y_train)

# Simplified plot_tree function
plt.figure(figsize=(10, 6))
plot_tree(clf_id3, 
          filled=True, 
          feature_names=encoder.get_feature_names_out(),
          class_names=clf_id3.classes_,
          max_depth=3,
          fontsize=10)
plt.tight_layout()
plt.show()

y_pred_id3 = clf_id3.predict(X_test)
print("ID3 Algorithm Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_id3)}")
