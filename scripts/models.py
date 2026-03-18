# import pandas as pd
# import joblib
# from sklearn.model_selection import train_test_split
# # USE cuML INSTEAD OF scikit-learn FOR GPU 
# import cuml 
# from cuml.ensemble import RandomForestClassifier as cuRF
# from cuml.linear_model import LogisticRegression as cuLR
# from sklearn.neural_network import MLPClassifier # MLP usually stays CPU/PyTorch
# from sklearn.metrics import classification_report

# # 1. Load the CLEANED data 
# df = pd.read_csv('cleaned_data_export_orange.csv')

# # 2. Split Features and Target 
# X = df.drop('Class', axis=1).astype('float32') # GPU likes float32
# y = df['Class'].astype('int32')

# # 3. 80/20 Train-Test Split 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # --- MODEL 1: GPU Random Forest --- [cite: 127, 128]
# print("Training GPU Random Forest...")
# rf = cuRF(n_estimators=200) # Runs on RTX 4060
# rf.fit(X_train, y_train)
# joblib.dump(rf, 'gpu_random_forest.pkl')

# # --- MODEL 2: GPU Logistic Regression --- [cite: 129, 130]
# print("Training GPU Logistic Regression...")
# lr = cuLR(max_iter=1000)
# lr.fit(X_train, y_train)
# joblib.dump(lr, 'gpu_logistic_regression.pkl')

# # --- MODEL 3: Neural Network --- 
# print("Training Neural Network (CPU Intensive)...")
# mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
# mlp.fit(X_train, y_train)

# # 4. Evaluation [cite: 123]
# # Note: cuML models predict on GPU; results move to CPU for the report
# print("\n--- RANDOM FOREST (GPU) EVALUATION ---")
# print(classification_report(y_test, rf.predict(X_test).to_numpy()))





import pandas as pd
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb

# -------------------------
# Check GPU
# -------------------------
print("GPU available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("cleaned_data_export_orange.csv")
X = df.drop("Class", axis=1).astype("float32")
y = df["Class"].astype("int32")

# -------------------------
# Train/Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# MODEL 1 — GPU Tree Model (XGBoost)
# -------------------------
print("Training GPU Tree Model...")

dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
dtest = xgb.DMatrix(X_test.values, label=y_test.values)

params = {
    "tree_method": "hist",
    "max_depth": 12,
    "learning_rate": 0.05,
    "objective": "multi:softprob",
    "num_class": len(y.unique()),
    "device": "cuda",
    "eval_metric": "mlogloss"
}

num_boost_round = 1000  # train thoroughly
bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)

# Save XGBoost model
bst.save_model("gpu_tree_model.json")

# Evaluation
pred_rf_prob = bst.predict(dtest)
pred_rf = pred_rf_prob.argmax(axis=1)
print("\n--- GPU TREE MODEL EVALUATION ---")
print(classification_report(y_test, pred_rf))

# -------------------------
# MODEL 2 — Logistic Regression (CPU)
# -------------------------
print("\nTraining Logistic Regression...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=5000, solver="saga")
lr.fit(X_train_scaled, y_train)

joblib.dump(lr, "logistic_regression.pkl")
pred_lr = lr.predict(X_test_scaled)
print("\n--- LOGISTIC REGRESSION EVALUATION ---")
print(classification_report(y_test, pred_lr))

# -------------------------
# MODEL 3 — Neural Network (PyTorch, GPU)
# -------------------------
print("\nTraining Neural Network on GPU...")

# Convert to torch tensors
X_train_t = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train.values, dtype=torch.long).to(device)
X_test_t = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test.values, dtype=torch.long).to(device)

# Use DataLoader for batch training
train_dataset = TensorDataset(X_train_t, y_train_t)
batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Bigger network
model = torch.nn.Sequential(
    torch.nn.Linear(X_train.shape[1], 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, len(y.unique()))
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = loss_fn(outputs, yb)
        loss.backward()
        optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save NN model
torch.save(model.state_dict(), "nn_model.pt")

# Evaluation
model.eval()
with torch.no_grad():
    outputs_test = model(X_test_t)
    pred_nn = torch.argmax(outputs_test, dim=1).cpu().numpy()

print("\n--- NEURAL NETWORK EVALUATION ---")
print(classification_report(y_test, pred_nn))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------------
# 1. VISUAL: Confusion Matrix for XGBoost (The Winner)
# -------------------------
print("Generating Confusion Matrix...")
cm = confusion_matrix(y_test, pred_rf)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legit', 'Fraud'])
disp.plot(cmap='Blues')
plt.title('XGBoost Fraud Detection: Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# -------------------------
# 2. VISUAL: Feature Importance (Data Mining Discovery)
# -------------------------
print("Generating Feature Importance Chart...")
# Get importance scores from XGBoost
importance = bst.get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'Feature': list(importance.keys()),
    'Importance': list(importance.values())
}).sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Top 10 Most Predictive Features for Fraud')
plt.savefig('feature_importance.png')
plt.show()
from mlxtend.frequent_patterns import apriori, association_rules

# We simplify the data for mining rules (focusing on high-impact features)
df_mining = df[['V14', 'V12', 'V10', 'Class']]

# Creating bins (High/Low) to make rules readable
for col in ['V14', 'V12', 'V10']:
    df_mining[col] = pd.qcut(df_mining[col], 2, labels=['Low', 'High'])

# Convert to dummy variables for Apriori
df_dummies = pd.get_dummies(df_mining)

# Mine the rules
frequent_itemsets = apriori(df_dummies, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

# Filter for rules that lead to Fraud
fraud_rules = rules[rules['consequents'].astype(str).str.contains('Class_1')]
print(fraud_rules[['antecedents', 'consequents', 'support', 'confidence']].head(5))