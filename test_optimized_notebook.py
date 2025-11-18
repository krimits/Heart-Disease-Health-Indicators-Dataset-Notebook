#!/usr/bin/env python3
"""
Test script για το βελτιστοποιημένο Cancer Prediction notebook
Ελέγχει τις βασικές λειτουργίες και dependencies
"""

import sys

def test_imports():
    """Έλεγχος ότι όλες οι απαραίτητες βιβλιοθήκες υπάρχουν"""
    print("🔍 Testing imports...")

    required_packages = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical operations',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical visualizations',
        'sklearn': 'Machine Learning',
        'joblib': 'Model persistence'
    }

    missing_packages = []

    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {package:15s} - {description}")
        except ImportError:
            print(f"   ❌ {package:15s} - {description} (MISSING)")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install " + " ".join(missing_packages))
        return False

    print("\n✅ All required packages are installed!")
    return True


def test_data_loading():
    """Έλεγχος ότι μπορούμε να φορτώσουμε τα δεδομένα"""
    print("\n🔍 Testing data loading...")

    try:
        import pandas as pd
        url = 'https://raw.githubusercontent.com/ybifoundation/Dataset/main/Cancer.csv'
        cancer = pd.read_csv(url)

        print(f"   ✅ Dataset loaded successfully!")
        print(f"   ✅ Shape: {cancer.shape}")
        print(f"   ✅ Columns: {len(cancer.columns)}")

        # Έλεγχος για expected columns
        assert 'diagnosis' in cancer.columns, "Missing 'diagnosis' column"
        assert 'id' in cancer.columns, "Missing 'id' column"

        print(f"   ✅ Expected columns present")
        return True

    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return False


def test_basic_preprocessing():
    """Έλεγχος βασικών preprocessing operations"""
    print("\n🔍 Testing basic preprocessing...")

    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # Load data
        url = 'https://raw.githubusercontent.com/ybifoundation/Dataset/main/Cancer.csv'
        cancer = pd.read_csv(url)

        # Preprocessing
        columns_to_drop = ['id', 'Unnamed: 32'] if 'Unnamed: 32' in cancer.columns else ['id']
        cancer_clean = cancer.drop(columns=columns_to_drop)

        X = cancer_clean.drop('diagnosis', axis=1)
        y = cancer_clean['diagnosis']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        print(f"   ✅ Train-test split: {X_train.shape[0]} train, {X_test.shape[0]} test")

        # Feature scaling (correct way - no data leakage!)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"   ✅ Feature scaling completed (no data leakage)")
        print(f"   ✅ Scaled train shape: {X_train_scaled.shape}")
        print(f"   ✅ Scaled test shape: {X_test_scaled.shape}")

        # Verify no data leakage
        train_mean = X_train_scaled.mean()
        test_mean = X_test_scaled.mean()
        print(f"   ✅ Train mean: {train_mean:.6f} (should be ~0)")
        print(f"   ✅ Test mean: {test_mean:.6f} (will differ slightly)")

        return True

    except Exception as e:
        print(f"   ❌ Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_training():
    """Έλεγχος ότι μπορούμε να εκπαιδεύσουμε μοντέλα"""
    print("\n🔍 Testing model training...")

    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        # Load and preprocess data
        url = 'https://raw.githubusercontent.com/ybifoundation/Dataset/main/Cancer.csv'
        cancer = pd.read_csv(url)
        columns_to_drop = ['id', 'Unnamed: 32'] if 'Unnamed: 32' in cancer.columns else ['id']
        cancer_clean = cancer.drop(columns=columns_to_drop)

        X = cancer_clean.drop('diagnosis', axis=1)
        y = cancer_clean['diagnosis']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Test Logistic Regression
        print("   📊 Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=5000, random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        print(f"   ✅ Logistic Regression Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")

        # Test Random Forest
        print("   📊 Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        print(f"   ✅ Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")

        # Verify reasonable accuracy (should be > 90%)
        assert lr_accuracy > 0.90, f"LR accuracy too low: {lr_accuracy}"
        assert rf_accuracy > 0.90, f"RF accuracy too low: {rf_accuracy}"

        print(f"   ✅ Both models achieve >90% accuracy")
        return True

    except Exception as e:
        print(f"   ❌ Error in model training: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_persistence():
    """Έλεγχος model saving/loading"""
    print("\n🔍 Testing model persistence...")

    try:
        import pandas as pd
        import joblib
        import os
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        # Load and preprocess data
        url = 'https://raw.githubusercontent.com/ybifoundation/Dataset/main/Cancer.csv'
        cancer = pd.read_csv(url)
        columns_to_drop = ['id', 'Unnamed: 32'] if 'Unnamed: 32' in cancer.columns else ['id']
        cancer_clean = cancer.drop(columns=columns_to_drop)

        X = cancer_clean.drop('diagnosis', axis=1)
        y = cancer_clean['diagnosis']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = LogisticRegression(max_iter=5000, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Get original predictions
        original_pred = model.predict(X_test_scaled)
        original_accuracy = accuracy_score(y_test, original_pred)

        # Save model and scaler
        test_model_file = 'test_model.pkl'
        test_scaler_file = 'test_scaler.pkl'

        joblib.dump(model, test_model_file)
        joblib.dump(scaler, test_scaler_file)
        print(f"   ✅ Model saved to {test_model_file}")
        print(f"   ✅ Scaler saved to {test_scaler_file}")

        # Load model and scaler
        loaded_model = joblib.load(test_model_file)
        loaded_scaler = joblib.load(test_scaler_file)
        print(f"   ✅ Model loaded from {test_model_file}")
        print(f"   ✅ Scaler loaded from {test_scaler_file}")

        # Test loaded model
        loaded_pred = loaded_model.predict(X_test_scaled)
        loaded_accuracy = accuracy_score(y_test, loaded_pred)

        # Verify predictions are identical
        assert (original_pred == loaded_pred).all(), "Loaded model predictions differ!"
        assert original_accuracy == loaded_accuracy, "Accuracy differs!"

        print(f"   ✅ Loaded model accuracy: {loaded_accuracy:.4f}")
        print(f"   ✅ Predictions match original model")

        # Cleanup
        os.remove(test_model_file)
        os.remove(test_scaler_file)
        print(f"   ✅ Cleanup completed")

        return True

    except Exception as e:
        print(f"   ❌ Error in model persistence: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Κύρια συνάρτηση test"""
    print("="*80)
    print("🧪 TESTING OPTIMIZED CANCER PREDICTION NOTEBOOK")
    print("="*80)

    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Preprocessing", test_basic_preprocessing),
        ("Model Training", test_model_training),
        ("Model Persistence", test_model_persistence)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status} - {test_name}")

    print("\n" + "="*80)
    print(f"🎯 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*80)

    if passed == total:
        print("\n✅ All tests passed! The optimized notebook is ready to use.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
