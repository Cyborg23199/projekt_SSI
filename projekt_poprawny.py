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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# 1. WLASNA IMPLEMENTACJA - NAIVE BAYES
# =======================

class NaiveBayesCustom:
    """ Własna implementacja Naive Bayes dla klasyfikacji binarnej """
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
            self.var[idx, :] = X_c.var(axis=0) + 1e-9
            self.priors[idx] = len(X_c) / len(X)
        return self

    def _calculate_likelihood(self, X, mean, var):
        numerator = np.exp(-(X - mean) ** 2 / (2 * var + 1e-9))
        denominator = np.sqrt(2 * np.pi * var + 1e-9)
        return numerator / denominator

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
            if np.any(np.isinf(posteriors)):
                posteriors = np.where(np.isinf(posteriors), 1e10, posteriors)
            posteriors = np.exp(posteriors - np.max(posteriors))
            posterior_sum = posteriors.sum()
            if posterior_sum > 0:
                posteriors = posteriors / posterior_sum
            else:
                posteriors = np.ones_like(posteriors) / len(posteriors)
            probas.append(posteriors)
        return np.array(probas)

# =======================
# 2. WLASNA IMPLEMENTACJA - DECISION TREE
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
        self.feature_importances_ = None
        self.n_features = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features)
        self.tree = self._build_tree(X, y, depth=0)
        # Normalizuj importances
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ = self.feature_importances_ / self.feature_importances_.sum()
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        best_gain = 0
        parent_gini = self._calculate_gini(y)
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            
            # Spróbuj percentile i unique values
            quantile_thresholds = np.percentile(feature_values, np.linspace(5, 95, 20))
            unique_thresholds = np.unique(feature_values)
            
            if len(unique_thresholds) > 50:
                unique_thresholds = np.random.choice(unique_thresholds, 50, replace=False)
            
            all_thresholds = np.unique(np.concatenate([quantile_thresholds, unique_thresholds]))
            
            for threshold in all_thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                if n_left < 1 or n_right < 1:
                    continue
                
                left_gini = self._calculate_gini(y[left_mask])
                right_gini = self._calculate_gini(y[right_mask])
                
                weighted_child_gini = (n_left / n_samples) * left_gini + (n_right / n_samples) * right_gini
                gain = parent_gini - weighted_child_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_gini = weighted_child_gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # KLUCZOWE: Dodaj gain do feature importance
        self.feature_importances_[best_feature] += best_gain * n_samples / len(y)
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold,
                    left=left_subtree, right=right_subtree)

    def _calculate_gini(self, y):
        """Oblicz Gini dla zbioru"""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        gini = 1 - np.sum(proportions ** 2)
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
# 5. MACIERZE BŁĘDÓW
# =======================

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Macierz błędów - wersja sklearn ConfusionMatrixDisplay
    """
    cm = confusion_matrix(y_true, y_pred)
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=['Spadek (0)', 'Wzrost (1)']
    )
    
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f'Macierz Błędów - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=300)
    plt.close()
    
    return cm


def print_confusion_matrix_analysis(cm, model_name):
    """
    Wydrukuj szczegółową analizę macierzy błędów
    """
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    print(f"\n{'='*60}")
    print(f"MACIERZ BŁĘDÓW - {model_name}")
    print(f"{'='*60}")
    print(f"{'':20} Przewidziana 0   Przewidziana 1")
    print(f"{'Rzeczywista 0':20} {tn:6d}           {fp:6d}")
    print(f"{'Rzeczywista 1':20} {fn:6d}           {tp:6d}")
    print(f"{'='*60}")
    
    print(f"\n📊 INTERPRETACJA:")
    print(f"  TN (True Negative):   {tn:6d}  - poprawnie przewidziano spadek")
    print(f"  FP (False Positive):  {fp:6d}  - błędnie przewidziano wzrost")
    print(f"  FN (False Negative):  {fn:6d}  - błędnie przewidziano spadek")
    print(f"  TP (True Positive):   {tp:6d}  - poprawnie przewidziano wzrost")
    
    print(f"\n📈 METRYKI Z MACIERZY:")
    
    if (tp + fn) > 0:
        sensitivity = tp / (tp + fn)
        print(f"  Sensitivity/Recall:  {sensitivity:.4f}  - jaką część wzrostów znaleźliśmy?")
    
    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
        print(f"  Specificity:         {specificity:.4f}  - jaką część spadków znaleźliśmy?")
    
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
        print(f"  Precision:           {precision:.4f}  - % prawidłowych wzrostów")
    
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    print(f"  Accuracy:            {accuracy:.4f}  - % prawidłowych wszystkich")
    
    if (tp + fp + fn) > 0:
        f1 = 2 * tp / (2 * tp + fp + fn)
        print(f"  F1-Score:            {f1:.4f}  - balans precision i recall")

# =======================
# 6. WAŻNOŚĆ CECH
# =======================

def calculate_permutation_importance_all(model, X_test, y_test, feature_names, model_type):
    """
    Oblicz ważność cech dla każdego modelu
    """
    from sklearn.metrics import accuracy_score
    
    # Dla WSZYSTKICH Decision Trees (sklearn i custom)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        result_df = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': importances[indices],
            'importance_pct': (importances[indices] / importances.sum() * 100) if importances.sum() > 0 else importances[indices]
        })
        return result_df, "GINI/GAIN based"
    
    else:
        # Permutation importance dla Naive Bayes
        y_pred_baseline = model.predict(X_test)
        baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
        
        importances = []
        
        for i in range(X_test.shape[1]):
            X_test_copy = X_test.copy()
            np.random.seed(42)
            np.random.shuffle(X_test_copy[:, i])
            
            y_pred_shuffled = model.predict(X_test_copy)
            shuffled_accuracy = accuracy_score(y_test, y_pred_shuffled)
            
            importance = baseline_accuracy - shuffled_accuracy
            importances.append(importance)
        
        result_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        total_imp = result_df['importance'].sum()
        if total_imp > 0:
            result_df['importance_pct'] = (result_df['importance'] / total_imp * 100)
        else:
            result_df['importance_pct'] = 0
        
        return result_df, "PERMUTATION"


def plot_feature_importance(importance_df, model_name):
    """
    Wizualizuj ważność cech
    """
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    bars = plt.barh(importance_df['feature'], importance_df['importance'], color=colors)
    plt.xlabel('Ważność', fontsize=12)
    plt.title(f'Ważność Cech - {model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.replace(" ", "_")}.png', dpi=300)
    plt.close()

# =======================
# 7. TRENING MODELI
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
# 8. MAIN
# =======================

def main():
    print("=" * 80)
    print("SYSTEM WSPOMAGANIA DECYZJI W FINANSACH")
    print("Analiza Trendów Giełdowych - Porównanie Metod")
    print("=" * 80)
    print()

    print("[1/5] Ładowanie wielu plików (.txt) jako akcje/ETF...")
    data_path = "archive/Stocks/*.txt"
    df = load_and_prepare_multi_stock_files(data_path)
    print(f"✓ Załadowano {len(df)} rekordów z {df['Ticker'].nunique()} tickerów")
    print(f"  Kolumny: {df.columns.tolist()}")
    print()

    print("[2/5] Pobieranie top tickerów...")
    top_100_tickers, df_filtered = get_top_tickers(df, n=100, start_year=1970, end_year=2017)
    print(f"✓ Wybrano {len(top_100_tickers)} tickerów, przykłady: {top_100_tickers[:10]}\n")

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

    print("[4/5] Przygotowywanie danych do treningu...")
    feature_cols = ['daily_return', 'sma_5', 'sma_20', 'rsi',
                    'volume_change', 'volatility', 'price_range', 'macd']
    data_clean = combined_data[feature_cols + ['target']].replace([np.inf, -np.inf], np.nan).dropna()

    X = data_clean[feature_cols].values
    y = data_clean['target'].values

    print(f"✓ Przygotowano {len(X)} próbek")
    print(f"  Rozkład klas: {np.bincount(y)}")
    print(f"  Cechy: {feature_cols}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Podzielono dane: {len(X_train)} train, {len(X_test)} test\n")

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
    model_dt_custom = DecisionTreeCustom(max_depth=11, min_samples_split=3)
    result4, _, _ = train_and_evaluate_model(
        X_train, X_test, y_train, y_test,
        "Decision Tree (Custom)", model_dt_custom
    )
    # Upewnij się że model ma feature_importances_
    assert hasattr(model_dt_custom, 'feature_importances_'), "Custom model nie ma feature_importances_"

    results_list.append(result4)
    print(f"    ✓ Accuracy: {result4['accuracy']:.4f}")
    print()

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
    
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("=" * 80)
    print("✓ Wyniki zapisane do 'model_comparison_results.csv'")
    print()

    # ===== MACIERZE BŁĘDÓW I WAŻNOŚĆ CECH =====
    print("\n\n" + "="*80)
    print("ANALIZA MACIERZY BŁĘDÓW I WAŻNOŚCI CECH")
    print("="*80)
    
    all_predictions = {
        'Naive Bayes (sklearn)': model_nb_sklearn.predict(X_test_scaled),
        'Naive Bayes (Custom)': model_nb_custom.predict(X_test_scaled),
        'Decision Tree (sklearn)': model_dt_sklearn.predict(X_test),
        'Decision Tree (Custom)': model_dt_custom.predict(X_test)
    }
    
    all_models = {
        'Naive Bayes (sklearn)': model_nb_sklearn,
        'Naive Bayes (Custom)': model_nb_custom,
        'Decision Tree (sklearn)': model_dt_sklearn,
        'Decision Tree (Custom)': model_dt_custom
    }
    
    model_types = {
        'Naive Bayes (sklearn)': 'Naive Bayes',
        'Naive Bayes (Custom)': 'Naive Bayes',
        'Decision Tree (sklearn)': 'Decision Tree',
        'Decision Tree (Custom)': 'Custom'
    }
    
    all_results = {}
    
    for model_name, y_pred in all_predictions.items():
        print(f"\n\n{'#'*80}")
        print(f"# {model_name}")
        print(f"{'#'*80}")
        
        # Macierz błędów
        cm = plot_confusion_matrix(y_test, y_pred, model_name)
        print_confusion_matrix_analysis(cm, model_name)
        
        # Ważność cech
        print(f"\n\n{'='*60}")
        print(f"WAŻNOŚĆ CECH - {model_name}")
        print(f"{'='*60}\n")
        
        model = all_models[model_name]
        model_type = model_types[model_name]
        
        X_test_for_importance = X_test_scaled if 'Naive Bayes' in model_name else X_test
        
        importance_df, method = calculate_permutation_importance_all(
            model, X_test_for_importance, y_test, feature_cols, model_type
        )
        
        print(f"Metoda: {method}\n")
        print(importance_df.to_string(index=False))
        
        plot_feature_importance(importance_df, model_name)
        
        all_results[model_name] = {
            'confusion_matrix': cm,
            'feature_importance': importance_df
        }
    
    # Porównanie cech między modelami
    print(f"\n\n{'='*80}")
    print("PORÓWNANIE WAŻNOŚCI CECH MIĘDZY MODELAMI")
    print(f"{'='*80}\n")
    
    comparison_df = pd.DataFrame({'feature': feature_cols})
    
    for model_name, results in all_results.items():
        importance_df = results['feature_importance']
        importance_dict = dict(zip(importance_df['feature'], importance_df['importance']))
        comparison_df[model_name] = comparison_df['feature'].map(importance_dict)
    
    comparison_df['średnia'] = comparison_df.iloc[:, 1:].mean(axis=1)
    comparison_df = comparison_df.sort_values('średnia', ascending=False)
    
    print(comparison_df.to_string(index=False))
    
    print("\n✓ Wizualizacje zapisane jako PNG")
    print("✓ Confusion matrices i feature importance wygenerowane")
    print("="*80)

    return results_df

if __name__ == "__main__":
    results = main()