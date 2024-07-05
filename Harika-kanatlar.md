import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Iris veri setini yükle
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Veriyi eğitim ve test setlerine böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Lojistik regresyon modelini oluştur ve eğit
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yap
y_pred = model.predict(X_test)

# Modelin doğruluğunu hesapla
accuracy = accuracy_score(y_test, y_pred)
print(f"Modelin doğruluğu: {accuracy:.2f}")

# Bazı tahmin örnekleri göster
for i in range(5):
    print(f"Gerçek: {y_test[i]}, Tahmin: {y_pred[i]}")
