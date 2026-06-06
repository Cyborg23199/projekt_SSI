import pandas as pd
import numpy as np
import warnings
import glob
import os

warnings.filterwarnings('ignore')

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# =======================
# 1. WLASNA IMPLEMENTACJA - NAIVE BAYES
# (niezmienione)
# =======================

class NaiveBayesCustom:
    def __init__(self):
        self.mean = None
        self.var = None
        self.priors = None
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.mean = np.zeros((n_classes, X.shape[1]))
        self.var = np.zeros((n_classes, X.shape[1]))
        self.priors = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-9  # Dodaj epsilon
            self.priors[idx] = len(X_c) / len(X)
        
        return self

    def _calculate_likelihood(self, X, mean, var):
        numerator = np.exp(-(X - mean) ** 2 / (2 * var + 1e-9))
        denominator = np.sqrt(2 * np.pi * (var + 1e-9))
        return numerator / (denominator + 1e-9)

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[idx] + 1e-9)
                likelihood = np.log(self._calculate_likelihood(x, self.mean[idx, :], self.var[idx, :]) + 1e-9).sum()
                posterior = prior + likelihood
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)

    def predict_proba(self, X):
        probas = []
        for x in X:
            posteriors = []
            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[idx] + 1e-9)
                likelihood = np.log(self._calculate_likelihood(x, self.mean[idx, :], self.var[idx, :]) + 1e-9).sum()
                posterior = prior + likelihood
                posteriors.append(posterior)
            
            posteriors = np.array(posteriors)
            
            # Obsługa inf
            if np.any(np.isinf(posteriors)):
                posteriors = np.where(np.isinf(posteriors), 1e10, posteriors)
            
            # Softmax z numeryczną stabilnością
            posteriors = np.exp(posteriors - np.max(posteriors))
            posterior_sum = posteriors.sum()
            
            if posterior_sum > 0:
                posteriors = posteriors / posterior_sum
            else:
                posteriors = np.ones_like(posteriors) / len(posteriors)
            
            probas.append(posteriors)
        
        return np.array(probas)

# =======================
# 2. WLASNA IMPLEMENTACJA - DECISION TREE (niezmienione)
# =======================

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeCustom:
    def __init__(self, max_depth=5, min_samples_split=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Warunki zatrzymania
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        
        # Szukaj najlepszego podziału
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            
            # Użyj quantile-based thresholds (szybsze i lepsze)
            thresholds = np.percentile(feature_values, np.linspace(10, 90, 15))
            thresholds = np.unique(thresholds)
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                # Wymagaj minimum próbek
                if n_left < 2 or n_right < 2:
                    continue
                
                gini = self._gini_index(y, y[left_mask], y[right_mask])
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        # Jeśli nie znaleziono podziału
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Dokonaj podziału
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold,
                    left=left_subtree, right=right_subtree)

    def _gini_index(self, parent, left_child, right_child):
        n = len(parent)
        n_left = len(left_child)
        n_right = len(right_child)
        
        if n_left == 0 or n_right == 0:
            return float('inf')
        
        unique_classes = np.unique(parent)
        
        gini_left = 1.0 - sum((np.sum(left_child == c) / n_left) ** 2 
                              for c in unique_classes)
        gini_right = 1.0 - sum((np.sum(right_child == c) / n_right) ** 2 
                               for c in unique_classes)
        
        gini = (n_left / n) * gini_left + (n_right / n) * gini_right
        return gini

    def _most_common_label(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        return np.argmax(counts)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict_proba(self, X):
        predictions = self.predict(X)
        proba = np.zeros((len(X), 2))
        for i, pred in enumerate(predictions):
            if 0 <= pred <= 1:
                proba[i, int(pred)] = 1.0
        return proba

# =======================
# 3. FEATURE ENGINEERING
# =======================

def calculate_features(df):
    df = df.copy()
    df['daily_return'] = df['Adj Close'].pct_change()
    df['sma_5'] = df['Adj Close'].rolling(window=5).mean()
    df['sma_20'] = df['Adj Close'].rolling(window=20).mean()
    df['rsi'] = calculate_rsi(df['Adj Close'], period=14)
    df['volume_change'] = df['Volume'].pct_change()
    df['volatility'] = df['daily_return'].rolling(window=5).std()
    df['price_range'] = (df['High'] - df['Low']) / df['Close']
    df['macd'] = calculate_macd(df['Adj Close'])
    df['target'] = (df['Adj Close'].shift(-1) > df['Adj Close']).astype(int)
    return df

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd

# =======================
# 4. PRZYGOTOWANIE DANYCH
# =======================

def load_and_prepare_multi_stock_files(data_path_pattern):
    files = glob.glob(data_path_pattern)
    all_data = []
    print(f"Znaleziono {len(files)} plików...")
    for f in files:
        try:
            ticker = os.path.basename(f).split(".")[0].upper()
            df = pd.read_csv(f)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            # Ujednolicenie kolumn ("Adj Close" dojedzie na końcu jeśli nie ma)
            if 'Adj Close' not in df.columns and 'Close' in df.columns:
                df['Adj Close'] = df['Close']
            df['Ticker'] = ticker
            all_data.append(df)
        except Exception as e:
            print(f"Błąd przy {f}: {e}")
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def get_top_tickers(df, n=100, start_year=1970, end_year=9999):
    df_filtered = df[(df['Date'].dt.year >= start_year) & (df['Date'].dt.year <= end_year)].copy()
    ticker_volumes = df_filtered.groupby('Ticker')['Volume'].sum().sort_values(ascending=False)
    top_tickers = ticker_volumes.head(n).index.tolist()
    return top_tickers, df_filtered

# =======================
# 5. TRENING MODELI
# =======================

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] > 1:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = np.nan
        else:
            roc_auc = np.nan
    except:
        roc_auc = np.nan
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    return results, model, y_pred

# =======================
# 6. MAIN
# =======================

def main():
    print("=" * 80)
    print("SYSTEM WSPOMAGANIA DECYZJI W FINANSACH")
    print("Analiza Trendów Giełdowych - Porównanie Metod")
    print("=" * 80)
    print()

    # KROK 1: Ładuj z plików!
    print("[1/5] Ładowanie wielu plików (.txt) jako akcje/ETF...")
    data_path = "archive/Stocks/*.txt"  # <- Twój katalog!
    df = load_and_prepare_multi_stock_files(data_path)
    print(f"✓ Załadowano {len(df)} rekordów z {df['Ticker'].nunique()} tickerów")
    print(f"  Kolumny: {df.columns.tolist()}")
    print()

    # KROK 2: top tickers i filtr dat
    print("[2/5] Pobieranie top tickerów...")
    top_100_tickers, df_filtered = get_top_tickers(df, n=100, start_year=1970, end_year=2017)
    print(f"✓ Wybrano {len(top_100_tickers)} tickerów, przykłady: {top_100_tickers[:10]}\n")

    # KROK 3: Oblicz cechy
    print("[3/5] Obliczanie cech (features engineering)...")
    all_data = []
    valid_tickers = []
    for ticker in top_100_tickers:
        ticker_data = df_filtered[df_filtered['Ticker'] == ticker].copy()
        if len(ticker_data) < 50:
            continue
        ticker_data = calculate_features(ticker_data)
        all_data.append(ticker_data)
        valid_tickers.append(ticker)
        if len(valid_tickers) % 10 == 0:
            print(f"  Przetworzono {len(valid_tickers)} tickerów...")
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"✓ Obliczono cechy dla {len(valid_tickers)} tickerów")
    print(f"  Całkowicie rekordów: {len(combined_data)}\n")

    # KROK 4: Przygotuj dane do treningu
    print("[4/5] Przygotowywanie danych do treningu...")
    feature_cols = ['daily_return', 'sma_5', 'sma_20', 'rsi',
                    'volume_change', 'volatility', 'price_range', 'macd']
    # USUŃ inf i NaN:
    data_clean = combined_data[feature_cols + ['target']].replace([np.inf, -np.inf], np.nan).dropna()

    X = data_clean[feature_cols].values
    y = data_clean['target'].values

    print(f"✓ Przygotowano {len(X)} próbek")
    print(f"  Rozkład klas: {np.bincount(y)}")
    print(f"  Cechy: {feature_cols}\n")

    print("NaN w X:", np.isnan(X).sum())
    print("Inf w X:", np.isinf(X).sum())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Podzielono dane: {len(X_train)} train, {len(X_test)} test\n")

    # KROK 5: Trenuj i ewaluuj modele
    print("[5/5] Trenowanie i ewaluacja modeli...\n")
    results_list = []
    print("  [1/4] Naive Bayes (sklearn)...")
    model_nb_sklearn = GaussianNB()
    result1, _, _ = train_and_evaluate_model(
        X_train_scaled, X_test_scaled, y_train, y_test,
        "Naive Bayes (sklearn)", model_nb_sklearn
    )
    results_list.append(result1)
    print(f"    ✓ Accuracy: {result1['accuracy']:.4f}")
    print("  [2/4] Naive Bayes (Custom)...")
    model_nb_custom = NaiveBayesCustom()
    result2, _, _ = train_and_evaluate_model(
        X_train_scaled, X_test_scaled, y_train, y_test,
        "Naive Bayes (Custom)", model_nb_custom
    )
    results_list.append(result2)
    print(f"    ✓ Accuracy: {result2['accuracy']:.4f}")
    print("  [3/4] Decision Tree (sklearn)...")
    model_dt_sklearn = DecisionTreeClassifier(max_depth=10, random_state=42)
    result3, _, _ = train_and_evaluate_model(
        X_train, X_test, y_train, y_test,
        "Decision Tree (sklearn)", model_dt_sklearn
    )
    results_list.append(result3)
    print(f"    ✓ Accuracy: {result3['accuracy']:.4f}")
    print("  [4/4] Decision Tree (Custom)...")
    model_dt_custom = DecisionTreeCustom(max_depth=5, min_samples_split=20)
    result4, _, _ = train_and_evaluate_model(
        X_train, X_test, y_train, y_test,
        "Decision Tree (Custom)", model_dt_custom
    )
    results_list.append(result4)
    print(f"    ✓ Accuracy: {result4['accuracy']:.4f}")
    print()

    # Wyniki końcowe
    print("=" * 80)
    print("PODSUMOWANIE WYNIKÓW")
    print("=" * 80)
    results_df = pd.DataFrame(results_list)
    print(results_df.to_string(index=False))
    print()
    print("-" * 80)
    print("PORÓWNANIE ACCURACY")
    print("-" * 80)
    accuracy_comparison = results_df[['model_name', 'accuracy']].sort_values('accuracy', ascending=False)
    for idx, row in accuracy_comparison.iterrows():
        bar = "█" * int(row['accuracy'] * 50)
        print(f"{row['model_name']:<30} {row['accuracy']:.4f} {bar}")
    print()
    print("-" * 80)
    print("PORÓWNANIE F1-SCORE")
    print("-" * 80)
    f1_comparison = results_df[['model_name', 'f1']].sort_values('f1', ascending=False)
    for idx, row in f1_comparison.iterrows():
        bar = "█" * int(row['f1'] * 50)
        print(f"{row['model_name']:<30} {row['f1']:.4f} {bar}")
    print()
    best_model_idx = results_df['accuracy'].idxmax()
    best_model = results_df.iloc[best_model_idx]
    print("-" * 80)
    print("NAJLEPSZY MODEL")
    print("-" * 80)
    print(f"Model: {best_model['model_name']}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"Precision: {best_model['precision']:.4f}")
    print(f"Recall: {best_model['recall']:.4f}")
    print(f"F1-Score: {best_model['f1']:.4f}")
    print()
    print("=" * 80)
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\n✓ Wyniki zapisane do 'model_comparison_results.csv'")

    return results_df

if __name__ == "__main__":
    results = main()