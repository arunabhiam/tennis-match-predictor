import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("cleaned_tennis_with_form.csv")
df = df.dropna()

# Feature engineering
df['rank_diff'] = df['Rank_1'] - df['Rank_2']
df['is_p1_higher'] = (df['Rank_1'] < df['Rank_2']).astype(int)
df['odds_ratio'] = abs(df['Odd_2'] / (df['Odd_1'] + 1e-5))
df['rank_ratio'] = df['Rank_2'] / (df['Rank_1'] + 1e-5)
df['is_top10_match'] = ((df['Rank_1'] <= 10) & (df['Rank_2'] <= 10)).astype(int)
df['surface_win_diff'] = df['surface_win_pct_p1'] - df['surface_win_pct_p2']


df = df[(df['Odd_1'] > 0) & (df['Odd_2'] > 0)]


feature_cols = [
    'odds_ratio','rank_diff', 'is_p1_higher', 'rank_ratio',
    'surface_win_diff','Surface_Clay', 'Surface_Grass', 'Surface_Hard', 'Surface_Carpet',
    'form_win_pct_diff','h2h_diff','is_top10_match'
]

X = df[feature_cols]
y = df['label']  # 1 if Player_1 won, 0 if Player_2 won

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# XGBoost model
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1

results = pd.DataFrame(X_test, columns=feature_cols)
results['actual'] = y_test.values
results['predicted'] = y_pred
results['confidence'] = y_proba

results.to_csv("test_predictions_with_confidence.csv", index=False)
print("Saved predictions with confidence to test_predictions_with_confidence.csv")

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model & scaler saved as model.pkl and scaler.pkl")

'''import matplotlib.pyplot as plt
from xgboost import plot_importance
plot_importance(model, max_num_features=15, importance_type='gain')
plt.show()
'''

