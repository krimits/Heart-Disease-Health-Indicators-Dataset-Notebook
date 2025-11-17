# Cancer Prediction – Σχολιασμένο Notebook

## Σκοπός

Σε αυτό το notebook χρησιμοποιούμε το **Breast Cancer Wisconsin Dataset** για να προβλέψουμε αν ένας όγκος είναι:
- `M` = Malignant (κακοήθης)
- `B` = Benign (καλοήθης)

Χρησιμοποιούμε διάφορους αλγορίθμους ταξινόμησης (Logistic Regression, KNN, SVC, RandomForest), συγκρίνουμε τις επιδόσεις τους και οπτικοποιούμε τα αποτελέσματα. Το notebook είναι δομημένο έτσι ώστε να μπορεί να χρησιμοποιηθεί στην τάξη, βήμα‑βήμα.

---

## Βήμα 1 – Εισαγωγή βιβλιοθηκών

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

pd.set_option('display.max_columns', 50)
pd.set_option('display.precision', 3)
```

**Τι κάνει:**

- Φορτώνει τις βασικές βιβλιοθήκες για:
  - **διαχείριση δεδομένων**: `pandas`, `numpy`
  - **οπτικοποίηση**: `matplotlib`, `seaborn`
  - **Machine Learning**: κλάσεις και συναρτήσεις από `sklearn` (διαχωρισμός train/test, κλίμακα, αλγόριθμοι, metrics).
- Οι `set_option` ρυθμίζουν πώς θα εμφανίζονται τα DataFrames (μέγιστες στήλες, ακρίβεια δεκαδικών).

**Διδακτικά:**  
Καλή στιγμή να εξηγήσεις τον ρόλο κάθε βιβλιοθήκης. Μπορείς να ζητήσεις από τους φοιτητές να εντοπίσουν αργότερα “πού” χρησιμοποιείται κάθε import.

---

## Βήμα 2 – Φόρτωση & αρχική εξερεύνηση δεδομένων

```python
cancer = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv')
cancer.head()
```

**Τι κάνει:**

- Κατεβάζει το dataset απευθείας από GitHub και το φορτώνει σε `DataFrame` με όνομα `cancer`.
- `head()` εμφανίζει τις πρώτες 5 γραμμές.

**Τι βλέπουμε:**

- Στήλες όπως:
  - `id`, `diagnosis` (στόχος), `radius_mean`, `texture_mean`, …, `fractal_dimension_worst`, `Unnamed: 32`.
- 33 στήλες συνολικά.

**Διδακτικά:**

- Εδώ εξηγείς ότι:
  - `diagnosis` είναι αυτό που θέλουμε να προβλέψουμε.
  - Οι υπόλοιπες αριθμητικές στήλες είναι χαρακτηριστικά (features) για τη μορφολογία των κυττάρων.

---

## Βήμα 3 – Πληροφορίες για τη δομή

```python
cancer.info()
```

**Τι κάνει:**

- Δείχνει:
  - πόσες γραμμές/στήλες,
  - τύπο δεδομένων κάθε στήλης,
  - πόσα μη‑κενά (Non-Null Count).

**Κρίσιμη παρατήρηση:**

- Όλες οι στήλες έχουν 569 μη‑κενές τιμές **εκτός** από `Unnamed: 32`, που έχει 0.
- Η `diagnosis` είναι `object` (string: M/B), σχεδόν όλες οι άλλες `float64`.

**Διδακτικά:**

- Τονίζεις τη σημασία του `info()` για:
  - εντοπισμό άχρηστων στηλών (π.χ. `Unnamed: 32`),
  - έλεγχο για missing values,
  - κατανόηση τύπων.

---

## Βήμα 4 – Περιγραφικά στατιστικά

```python
cancer.describe()
```

**Τι κάνει:**

- Δείχνει βασικά στατιστικά για **αριθμητικές** στήλες:
  - `count, mean, std, min, 25%, 50%, 75%, max`.

**Τι μάθουμε:**

- Κλίμακες των χαρακτηριστικών:
  - π.χ. `radius_mean` ~ [7, 28],
  - `area_mean` ~ [143, 2501],
  - `fractal_dimension_*` ~ [0.05, 0.2].
- Επιβεβαιώνουμε ότι `Unnamed: 32` έχει μόνο NaN.

**Διδακτικά:**

- Πλαίσιο για συζήτηση:
  - γιατί features σε τόσο διαφορετικές κλίμακες δημιουργούν ανάγκη για **scaling**.

---

## Βήμα 5 – Ονόματα στηλών

```python
cancer.columns
```

**Τι κάνει:**

- Τυπώνει μια λίστα με τα ονόματα όλων των στηλών.

**Διδακτικά:**

- Χρήσιμο για να αποφασίσουμε:
  - ποιο θα είναι το target,
  - ποιες στήλες θα αφαιρέσουμε (αναγνωριστικά, σκουπίδια).

---

## Βήμα 6 – Ορισμός του target (y) και των features (X)

```python
y = cancer['diagnosis']
X = cancer.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1)
X.shape, y.shape
```

**Τι κάνει:**

- `y`: παίρνουμε τη στήλη `diagnosis` ως target (M/B).
- `X`: αφαιρούμε:
  - `diagnosis` (γιατί είναι η απάντηση),
  - `id` (μόνο αναγνωριστικό),
  - `Unnamed: 32` (άδεια στήλη),
  και κρατάμε όλες τις υπόλοιπες 30 αριθμητικές στήλες ως features.

**Διδακτικά:**

- Τονίζεις τη βασική ιδέα:  
  **X = τι ξέρει το μοντέλο**  
  **y = τι θέλουμε να προβλέψει**
- Χρησιμοποίησε `X.shape` και `y.shape` για να δείξεις:
  - 569 δείγματα, 30 features.

---

## Βήμα 7 – Κατανομή κλάσεων

```python
y.value_counts()
```

**Τι κάνει:**

- Μετράει πόσες φορές εμφανίζεται κάθε τιμή του `diagnosis` (M/B).

**Γιατί έχει σημασία:**

- Για να δούμε αν το dataset είναι **ισορροπημένο** ή όχι:
  - συνήθως ~212 `M`, 357 `B`.

**Διδακτικά:**

- Μπορείς να ξεκινήσεις κουβέντα για:
  - τι σημαίνει class imbalance,
  - γιατί σε ακραίες περιπτώσεις η accuracy είναι παραπλανητική.

---

## Βήμα 8 – Encoding του target σε 0/1

```python
y = y.replace({'M': 1, 'B': 0})
y.value_counts()
```

**Τι κάνει:**

- Αντικαθιστά:
  - `M` → `1`,
  - `B` → `0`.

**Γιατί:**

- Απλοποιεί:
  - τον ορισμό της “θετικής” κλάσης (1 = καρκίνος),
  - τη χρήση των ML αλγορίθμων που χειρίζονται binary labels ως 0/1.

**Διδακτικά:**

- Καλή ευκαιρία να εξηγήσεις έννοιες:
  - “positive” vs “negative” class,
  - πώς συνδέονται με metrics όπως precision/recall.

---

## Βήμα 9 – Train / Test split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2529, stratify=y
)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

**Τι κάνει:**

- Χωρίζει τα δεδομένα σε:
  - 80% training set,
  - 20% test set.
- `random_state`: για reproducibility.
- `stratify=y`: κρατά ίδια αναλογία M/B και στα δύο σύνολα.

**Διδακτικά:**

- Τονίζεις:
  - γιατί **δεν** εκπαιδεύουμε και αξιολογούμε στο ίδιο σύνολο,
  - τι σημαίνει “unseen data”.

---

## Βήμα 10 – Standardization των features

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Τι κάνει:**

- `fit_transform` στο train: υπολογίζει mean & std ανά στήλη και εφαρμόζει standardization.
- `transform` στο test: χρησιμοποιεί τις ίδιες παραμέτρους (χωρίς `fit` ξανά).

**Γιατί χρειάζεται:**

- KNN, SVC, Logistic Regression επηρεάζονται από την κλίμακα.
- Με standardized features:
  - κάθε στήλη συνεισφέρει πιο δίκαια.

**Διδακτικά:**

- Σημαντικό να τονίσεις:
  - ΠΟΤΕ δεν κάνουμε `fit` στον scaler χρησιμοποιώντας test data (data leakage).

---

## Βήμα 11 – Logistic Regression

```python
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nConfusion Matrix (Logistic Regression):\n", confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report (Logistic Regression):\n", classification_report(y_test, y_pred_lr))
```

**Τι κάνει:**

- Εκπαιδεύει ένα μοντέλο Logistic Regression πάνω στα scaled δεδομένα.
- Κάνει προβλέψεις στο test set.
- Τυπώνει:
  - Accuracy,
  - Confusion Matrix,
  - Classification Report (precision, recall, f1).

**Διδακτικά:**

- Εξήγησε:
  - ότι logistic regression δίνει πιθανότητες για την κλάση 1,
  - τι σημαίνουν precision/recall/f1.

---

## Βήμα 12 – KNN

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\nConfusion Matrix (KNN):\n", confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report (KNN):\n", classification_report(y_test, y_pred_knn))
```

**Τι κάνει:**

- Εκπαιδεύει KNN (ουσιαστικά αποθηκεύει τα train samples).
- Για κάθε νέο δείγμα κοιτά τους 5 κοντινότερους (σε απόσταση) και παίρνει πλειοψηφία.

**Διδακτικά:**

- Ιδανικό για να εξηγήσεις:
  - distance-based learning,
  - γιατί είναι τόσο σημαντικό το scaling.

---

## Βήμα 13 – SVC (SVM Classifier)

```python
svc = SVC(kernel='rbf', probability=True, random_state=2529)
svc.fit(X_train_scaled, y_train)
y_pred_svc = svc.predict(X_test_scaled)

print("SVC Accuracy:", accuracy_score(y_test, y_pred_svc))
print("\nConfusion Matrix (SVC):\n", confusion_matrix(y_test, y_pred_svc))
print("\nClassification Report (SVC):\n", classification_report(y_test, y_pred_svc))
```

**Τι κάνει:**

- Εκπαιδεύει Support Vector Classifier με RBF kernel.
- Χτίζει ένα μη γραμμικό hyperplane που ξεχωρίζει τις κλάσεις.

**Διδακτικά:**

- Στα μικρά, καθαρά datasets όπως αυτό, SVC συχνά έχει πολύ υψηλή απόδοση.
- Καλή στιγμή για να αναφέρεις:
  - τη φιλοσοφία του SVM: μεγιστοποίηση margin, support vectors.

---

## Βήμα 14 – Random Forest

```python
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=2529
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nConfusion Matrix (Random Forest):\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))
```

**Τι κάνει:**

- Εκπαιδεύει 100 decision trees σε διαφορετικά τυχαία υποσύνολα του training set.
- Η πρόβλεψη είναι majority vote των δέντρων.

**Διδακτικά:**

- Εξήγηση του concept:
  - bagging,
  - randomness σε samples & features,
  - robust “out-of-the-box” μοντέλο.

---

## Βήμα 15 – Πίνακας accuracy ανά μοντέλο

```python
model_names = ["LogisticRegression", "KNN", "SVC", "RandomForest"]
accuracies = [
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_knn),
    accuracy_score(y_test, y_pred_svc),
    accuracy_score(y_test, y_pred_rf)
]

results_df = pd.DataFrame({
    "Model": model_names,
    "Accuracy": accuracies
})
results_df
```

**Τι κάνει:**

- Φτιάχνει ένα `DataFrame` με το όνομα κάθε μοντέλου και την accuracy του.

**Διδακτικά:**

- Βοηθά στην “numeric” σύγκριση,
- δείχνει τη λογική συλλογής metrics σε έναν συγκεντρωτικό πίνακα.

---

## Βήμα 16 – Bar chart σύγκρισης accuracy

```python
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Model", y="Accuracy", palette="viridis")

plt.ylim(0.8, 1.0)
plt.title("Σύγκριση Accuracy μεταξύ μοντέλων")
plt.ylabel("Accuracy")
plt.xlabel("Μοντέλο")

for index, row in results_df.iterrows():
    plt.text(index, row["Accuracy"] + 0.005, f"{row['Accuracy']:.3f}", ha='center')

plt.tight_layout()
plt.show()
```

**Τι κάνει:**

- Οπτικοποιεί τις accuracies ως bar chart με `seaborn`.

**Διδακτικά:**

- Εύκολη, οπτική σύγκριση μοντέλων.
- Συζήτηση:
  - “Ποιο είναι καλύτερο;”
  - “Είναι αρκετή η accuracy για να αποφασίσουμε;”

---

## Βήμα 17 – Συνάρτηση για Confusion Matrix heatmap

```python
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=['Predicted 0 (Benign)', 'Predicted 1 (Malignant)'],
        yticklabels=['Actual 0 (Benign)', 'Actual 1 (Malignant)']
    )
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()
```

**Τι κάνει:**

- Ορίζει μια helper function για να σχεδιάζουμε confusion matrix ως heatmap.

**Διδακτικά:**

- Πλήρως οπτικό εργαλείο για να δουν:
  - TP, TN, FP, FN.
- Χρήσιμο για συζήτηση ειδικά πάνω στα **false negatives**.

---

## Βήμα 18 – Confusion matrices για κάθε μοντέλο

```python
plot_confusion_matrix(y_test, y_pred_lr, "Confusion Matrix - Logistic Regression")
plot_confusion_matrix(y_test, y_pred_knn, "Confusion Matrix - KNN")
plot_confusion_matrix(y_test, y_pred_svc, "Confusion Matrix - SVC")
plot_confusion_matrix(y_test, y_pred_rf, "Confusion Matrix - Random Forest")
```

**Τι κάνει:**

- Σχεδιάζει 4 heatmaps – ένα για κάθε μοντέλο.

**Διδακτικά:**

- Ζήτα από τους φοιτητές να συγκρίνουν:
  - ποιο μοντέλο έχει λιγότερα **FN** (κάτω αριστερά),
  - ποιο έχει περισσότερα **FP** (πάνω δεξιά),
  - ποιο trade‑off είναι αποδεκτό σε ιατρικό περιβάλλον.

---

## Βήμα 19 – Cross‑validation με Pipelines (Accuracy)

```python
log_reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_neighbors=5))
])

svc_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(kernel='rbf'))
])

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=2529
)

models_cv = {
    "LogisticRegression": log_reg_pipeline,
    "KNN": knn_pipeline,
    "SVC": svc_pipeline,
    "RandomForest": rf_model
}

cv = 5
cv_results = {}

for name, model in models_cv.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_results[name] = scores
    print(f"{name} CV Accuracy per fold: {scores}")
    print(f"{name} CV Accuracy mean={scores.mean():.4f}, std={scores.std():.4f}\n")
```

**Τι κάνει:**

- Ορίζει `Pipeline`s ώστε:
  - σε κάθε fold του cross‑validation,
  - να γίνεται `fit` του scaler **μόνο στο training fold** και μετά `transform` στο test fold.
- Χρησιμοποιεί `cross_val_score` με `cv=5`.

**Διδακτικά:**

- Τονίζεις:
  - γιατί χρειαζόμαστε cross‑validation (σταθερότητα, αξιοπιστία της επίδοσης),
  - γιατί χρησιμοποιούμε `Pipeline` (αποφυγή data leakage).

---

## Βήμα 20 – Οπτικοποίηση mean CV accuracy

```python
cv_model_names = list(cv_results.keys())
cv_means = [cv_results[m].mean() for m in cv_model_names]
cv_stds = [cv_results[m].std() for m in cv_model_names]

cv_df = pd.DataFrame({
    "Model": cv_model_names,
    "MeanAccuracy": cv_means,
    "StdAccuracy": cv_stds
})

plt.figure(figsize=(8, 5))
sns.barplot(data=cv_df, x="Model", y="MeanAccuracy", palette="magma")
plt.ylim(0.8, 1.0)
plt.title(f"Mean CV Accuracy (cv={cv}) ανά μοντέλο")
plt.ylabel("Mean CV Accuracy")
plt.xlabel("Μοντέλο")
for index, row in cv_df.iterrows():
    plt.text(
        index,
        row["MeanAccuracy"] + 0.005,
        f"{row['MeanAccuracy']:.3f} ± {row['StdAccuracy']:.3f}",
        ha='center'
    )
plt.tight_layout()
plt.show()
```

**Τι κάνει:**

- Συγκεντρώνει mean & std της accuracy από το CV.
- Τα οπτικοποιεί σε bar chart.

**Διδακτικά:**

- Μπορείς να εξηγήσεις:
  - τι σημαίνει υψηλό mean & χαμηλό std,
  - ποιο μοντέλο είναι πιο “σταθερό”.

---

## Βήμα 21 – Cross‑validation με Recall για την κακοήθη κλάση (1)

```python
for name, model in models_cv.items():
    scores_recall = cross_val_score(model, X, y, cv=cv, scoring='recall')
    print(f"{name} CV Recall (class 1): mean={scores_recall.mean():.4f}, std={scores_recall.std():.4f}")
```

**Τι κάνει:**

- Αντί για `accuracy`, χρησιμοποιεί `scoring='recall'`.
- Εστιάζει στην κλάση 1 (κακοήθης).

**Διδακτικά:**

- Σημαντικό σημείο:
  - Στα ιατρικά προβλήματα, η **ανάκληση της positive class** (να μην μας ξεφύγουν οι κακοήθεις) είναι συχνά πιο κρίσιμη από την απλή accuracy.
- Συζήτηση:
  - ποιο μοντέλο είναι καλύτερο αν δώσουμε προτεραιότητα στο recall της κακοήθους κλάσης;
  - τι trade‑offs αποδεχόμαστε (περισσότερα false positives για λιγότερα false negatives;).

---

## Ιδέες για ασκήσεις φοιτητών

1. **Αλλαγή hyperparameters:**
   - Αλλάξτε `n_neighbors` στο KNN και δείτε πώς αλλάζει η απόδοση.
   - Αλλάξτε `C` και `gamma` στο SVC.

2. **ROC Curves & AUC:**
   - Προσθέστε ROC curves για Logistic Regression, SVC και RandomForest.
   - Συγκρίνετε AUC.

3. **Threshold Tuning:**
   - Για Logistic Regression / SVC: αλλάξτε το decision threshold από 0.5 σε 0.3 και δείτε τι γίνεται στο recall της κλάσης 1.

4. **Feature importance:**
   - Χρησιμοποιήστε `feature_importances_` από RandomForest για να δείτε ποια features είναι πιο σημαντικά.

Αυτό το markdown μπορεί να συνοδεύει το notebook ως “σημειώσεις διδάσκοντα” ή να δοθεί στους φοιτητές ως οδηγός μελέτης.
