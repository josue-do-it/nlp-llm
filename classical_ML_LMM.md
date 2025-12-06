# 1. Vectorize text using TF-IDF

```python
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)

y_train = np.array(train_labels)
y_val = np.array(val_labels)
y_test = np.array(test_labels)

print(f"TF-IDF feature matrix shape: {X_train.shape}")
```



# 🔍 What TF-IDF Does

## **TF-IDF (Term Frequency – Inverse Document Frequency)**  
A method to convert text into numerical vectors.

### **Term Frequency (TF)**  
How many times a word appears in a document.

### **Inverse Document Frequency (IDF)**  
Importance of a word based on how rare it is across documents.

### **Results**
- Frequent but non-informative words (e.g., *the, and, to*) get **low weight**.  
- Rare but meaningful words get **high weight**.

---

# ⚙️ Important Parameters

## ✔ **max_features=5000**
Keeps only the **5000 most important tokens**.  
→ Reduces dimensionality → faster and more memory-efficient model.

## ✔ **ngram_range=(1, 2)**
Uses:
- **unigrams**: `"good"`, `"movie"`, `"hate"`
- **bigrams**: `"very bad"`, `"not good"`, `"hate speech"`

➡ Crucial for polarization detection:  
`"not good"` ≠ `"good"`

---

# 🔁 fit_transform() vs transform()

### **fit_transform()**
- Learns vocabulary on **training data**
- Converts text into TF-IDF vectors

### **transform()**
- Applies the learned vocabulary to **validation** and **test**

❗ **Never fit on validation/test** → avoids **data leakage**.

---

# 📐 Example of Feature Matrix Shape

If `X_train.shape` returns:




It means:
- **10,000 documents**
- **5,000 TF-IDF features**

---

# ✅ Why Use Classical Machine Learning?

## ✔ **Fast**
Models like Logistic Regression and SVM train extremely quickly.  
Great for baselines.

## ✔ **Interpretable**
You can identify:
- which words have strong weights  
- which tokens contribute to polarized opinions  
- how the model makes decisions  

## ✔ **Strong Baseline**
TF-IDF + Linear SVM often reaches performance close to small Transformers.

---

# ❌ Limitations of Classical ML (Bag-of-Words)

These models:
- do **not** understand context  
- ignore sentence structure  
- do not capture meaning relationships  

Example:
- `"Apple"` (fruit) ≠ `"Apple"` (company)

They struggle with:
- irony  
- sarcasm  
- masked insults  
- deep contextualization  

---

# ✅ The Three Models You Will Test

## ⭐ 1. **Logistic Regression**
- Great for binary classification (0/1)  
- Fast  
- Easy to interpret  
- Very compatible with TF-IDF  
→ Often the **best classical model**.

---

## ⭐ 2. **Naive Bayes**
- Extremely fast  
- Probability-based  
- Assumes feature independence (not always true but effective)  
→ Often the best for **small or simple datasets**.

---

## ⭐ 3. **Linear SVM (Support Vector Machine)**
- Very powerful for text classification  
- Finds the best separating hyperplane between classes  
- Works extremely well with TF-IDF  
→ Often the **best baseline** for NLP tasks.

---
# 📊 Train and Evaluate Classical ML Models

```python
classical_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Linear SVM': LinearSVC(random_state=42, max_iter=1000)
}

classical_results = {}

for name, model in classical_models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}...")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    classical_results[name] = accuracy

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))
```

# 🎯 Simple Summary

| Step | What It Does |
|------|--------------|
| **TF-IDF** | Converts text → numeric feature vectors |
| **Classical ML** | Fast, simple, effective models for text classification |
| **LogReg / NB / SVM** | Three strong baseline classifiers for NLP |

