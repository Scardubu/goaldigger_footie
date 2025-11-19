import os

import joblib

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    class MockMLFlow:
        @staticmethod
        def set_tracking_uri(*args, **kwargs): pass
        @staticmethod
        def set_experiment(*args, **kwargs): pass
        @staticmethod
        def start_run(*args, **kwargs): return MockMLFlow()
        @staticmethod
        def log_metric(*args, **kwargs): pass
        @staticmethod
        def log_param(*args, **kwargs): pass
        @staticmethod
        def sklearn_log_model(*args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    mlflow = MockMLFlow()
    mlflow.sklearn = MockMLFlow()
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, log_loss)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     cross_validate)

try:
    from imblearn.over_sampling import SMOTE
except ImportError as e:
    raise ImportError("imblearn (imbalanced-learn) is not installed or not found in the current environment. Please check your Python environment and install with 'pip install imbalanced-learn'.") from e
try:
    from ignite.engine import (Events, create_supervised_evaluator,
                               create_supervised_trainer)
    from ignite.handlers import EarlyStopping, ModelCheckpoint
    from ignite.metrics import Loss
except ImportError as e:
    raise ImportError("pytorch-ignite is not installed or not found in the current environment. Please check your Python environment and install with 'pip install pytorch-ignite'.") from e
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier


class FootballDataset(Dataset):
    """
    PyTorch Dataset that applies both basic and advanced feature engineering,
    and prepares data for focal or standard loss training.
    """
    def __init__(self, df: pd.DataFrame, use_focal_loss: bool = False):
        self.use_focal_loss = use_focal_loss
        df = self._preprocess(df)
        df = self._add_advanced_features(df)
        self.X = df.drop(columns=["target"]).values.astype(np.float32)
        self.y = df["target"].values.astype(np.int64)

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop missing crucial fields
        return df.dropna(subset=["team_perf", "opp_perf", "target"])

    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Rolling form over last 5 matches
        df = df.copy()
        df["form_5"] = df.groupby("team")["target"].transform(
            lambda x: x.rolling(5, min_periods=1).mean().fillna(0)
        )
        # Opponent-adjusted performance
        df["opp_perf_diff"] = df["team_perf"] - df["opp_perf"]
        # Manager change in last 90 days flag
        if "manager_change_date" in df.columns:
            df["mgr_change_recent"] = (
                pd.to_datetime(df["manager_change_date"]) >= (pd.Timestamp.now() - pd.Timedelta(days=90))
            ).astype(int)
        else:
            df["mgr_change_recent"] = 0
        return df.fillna(0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


class SimpleMLP(nn.Module):
    """
    A basic MLP to use as a stacking learner.
    """
    def __init__(self, input_dim: int):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.layers(x)


class FocalLoss(nn.Module):
    """
    Implements focal loss for classification to handle class imbalance.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal


def build_pipeline(config: dict):
    """
    Builds a scikit-learn pipeline that includes feature preprocessing and a model.
    This is mainly used for hyperparameter optimization in tune.py.
    """
    # For now, just return the stacking model
    return build_stacking_model(config)


def build_stacking_model(config: dict):
    """
    Constructs a scikit-learn StackingClassifier based on config.
    """
    learners = []
    # XGBoost parameters with proper nested configuration access
    xgb_params = config.get("xgboost", {}).get("hyperparameters", {})
    eval_metric = config.get("training", {}).get("eval_metric", "logloss")
    learners.append((
        "xgb", XGBClassifier(
            use_label_encoder=False,
            eval_metric=eval_metric,
            **xgb_params
        )
    ))
    # LightGBM parameters with proper nested configuration access
    lgb_params = config.get("lightgbm", {}).get("hyperparameters", {})
    learners.append(("lgbm", LGBMClassifier(**lgb_params)))

    # Use a smaller number of folds for small datasets
    # If dataset is small, use 2-fold or simple holdout validation
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    
    stack = StackingClassifier(
        estimators=learners,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=cv,
        passthrough=True
    )
    return stack


def train(config_path: str):
    # Load and validate config
    with open(config_path, "r") as f:
        full_cfg = yaml.safe_load(f)
    cfg = full_cfg.get("models", {})  # Use 'models' key for model config
    
    # Set default num_epochs if not specified
    if "training" not in cfg:
        cfg["training"] = {}
    if "num_epochs" not in cfg["training"]:
        cfg["training"]["num_epochs"] = 50  # Default value

    # MLflow setup
    mlflow_config = full_cfg.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri", "http://localhost:5000")
    experiment_name = mlflow_config.get("experiment_name", "FootballInsights")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        # Handle paths properly
        paths_config = full_cfg.get("paths", {}).get("data", {})
        raw_path = paths_config.get("raw_path", "./data/raw")
        cache_path = paths_config.get("cache_path", "./data/cache")
        reference_path = "./data/reference"  # Default reference path
        
        # Load or preprocess data
        raw_file = os.path.join(raw_path, "matches.csv")
        reference_file = os.path.join(reference_path, "valid_matches.csv")
        cache_file = os.path.join(cache_path, "preprocessed.pkl")

        # Prefer reference file if it exists
        if os.path.exists(reference_file):
            raw = pd.read_csv(reference_file)
            from scripts.ingest_data import harmonize_valid_matches
            raw = harmonize_valid_matches(raw)
        elif os.path.exists(raw_file):
            raw = pd.read_csv(raw_file)
        else:
            raise FileNotFoundError(f"No data file found at {reference_file} or {raw_file}")
        
        if os.path.exists(cache_file):
            df = joblib.load(cache_file)
        else:
            from scripts.ingest_data import preprocess
            df = preprocess(raw, full_cfg)
            os.makedirs(cache_path, exist_ok=True)
            joblib.dump(df, cache_file)

        # Imbalance handling choice
        class_balance_config = full_cfg.get("class_balance", {})
        use_focal = class_balance_config.get("use_focal_loss", True)
        
        if not use_focal:
            try:
                sm = SMOTE()
                X_sm, y_sm = sm.fit_resample(df.drop(columns=["target"]), df["target"])
                df = pd.concat([X_sm, pd.Series(y_sm, name="target")], axis=1)
            except Exception as e:
                print(f"Warning: SMOTE failed ({e}), continuing with original data")
                use_focal = True  # Fall back to focal loss

        # Feature/target split
        X = df.drop(columns=["target"])
        y = df["target"]
        
        # Clean data before model training: handle problematic columns
        print(f"Original feature dataframe shape: {X.shape}")
        
        # Remove non-numeric columns that aren't needed for training
        columns_to_drop = []
        for col in X.columns:
            if X[col].dtype == 'object':
                if col in ['match_id', 'league', 'home_team', 'away_team', 'team', 'home_team_id', 'away_team_id',
                           'match_date', 'manager_change_date']:
                    columns_to_drop.append(col)
                    print(f"Dropping non-numeric column: {col}")
                else:
                    # Try to convert string columns to numeric if possible
                    try:
                        X[col] = pd.to_numeric(X[col])
                        print(f"Converted column {col} to numeric")
                    except:
                        columns_to_drop.append(col)
                        print(f"Dropping non-convertible column: {col}")
        
        X = X.drop(columns=columns_to_drop)
        print(f"Cleaned feature dataframe shape: {X.shape}")
        
        # Ensure all remaining columns are numeric
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                raise ValueError(f"Non-numeric column {col} with dtype {X[col].dtype} remains after cleaning")

        # Split data for training and validation with proper stratification
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                             train_test_split)

        # Use stratified split to maintain class distribution
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
        print(f"Training class distribution: {y_train.value_counts().to_dict()}")
        print(f"Validation class distribution: {y_val.value_counts().to_dict()}")

        # Train stacking model
        model = build_stacking_model(cfg)

        # Check if the dataset is too small for proper training
        if X.shape[0] <= 30:  # Increased threshold for better validation
            print(f"WARNING: Small dataset detected ({X.shape[0]} samples). Using simplified training approach with cross-validation.")

            # Use cross-validation even for small datasets to get better estimates
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_validate

            # Simple model that works well with small data
            xgb_model = XGBClassifier(
                use_label_encoder=False,
                eval_metric=cfg.get("training", {}).get("eval_metric", "logloss"),
                early_stopping_rounds=None,  # Disable early stopping for small datasets
                n_estimators=50,  # Reduce complexity for small datasets
                max_depth=3,      # Prevent overfitting
                **cfg.get("xgboost", {}).get("hyperparameters", {})
            )

            # Train a robust model that works well with small data
            rf_model = RandomForestClassifier(
                n_estimators=50,  # Reduced for small datasets
                random_state=42,
                min_samples_leaf=2,  # Increased to prevent overfitting
                min_samples_split=5,  # Increased to prevent overfitting
                max_features='sqrt',
                max_depth=5       # Limit depth to prevent overfitting
            )

            # Perform cross-validation to get realistic performance estimates
            cv_folds = min(3, X.shape[0] // 5)  # Ensure at least 5 samples per fold
            if cv_folds < 2:
                cv_folds = 2

            print(f"Performing {cv_folds}-fold cross-validation...")

            # Cross-validate Random Forest
            rf_cv_scores = cross_validate(
                rf_model, X, y, cv=cv_folds,
                scoring=['accuracy', 'neg_log_loss'],
                return_train_score=True
            )

            # Cross-validate XGBoost
            xgb_cv_scores = cross_validate(
                xgb_model, X, y, cv=cv_folds,
                scoring=['accuracy', 'neg_log_loss'],
                return_train_score=True
            )

            # Log cross-validation results
            print("\n=== Cross-Validation Results ===")
            print(f"Random Forest - CV Accuracy: {rf_cv_scores['test_accuracy'].mean():.4f} (+/- {rf_cv_scores['test_accuracy'].std() * 2:.4f})")
            print(f"Random Forest - CV Log Loss: {-rf_cv_scores['test_neg_log_loss'].mean():.4f} (+/- {rf_cv_scores['test_neg_log_loss'].std() * 2:.4f})")
            print(f"XGBoost - CV Accuracy: {xgb_cv_scores['test_accuracy'].mean():.4f} (+/- {xgb_cv_scores['test_accuracy'].std() * 2:.4f})")
            print(f"XGBoost - CV Log Loss: {-xgb_cv_scores['test_neg_log_loss'].mean():.4f} (+/- {xgb_cv_scores['test_neg_log_loss'].std() * 2:.4f})")

            # Train final models on training data and evaluate on validation set
            print("\nTraining final models...")
            rf_model.fit(X_train, y_train)
            xgb_model.fit(X_train, y_train)

            # Evaluate on validation set (NOT training set)
            rf_val_pred = rf_model.predict(X_val)
            rf_val_proba = rf_model.predict_proba(X_val)
            rf_val_acc = accuracy_score(y_val, rf_val_pred)
            rf_val_logloss = log_loss(y_val, rf_val_proba)

            xgb_val_pred = xgb_model.predict(X_val)
            xgb_val_proba = xgb_model.predict_proba(X_val)
            xgb_val_acc = accuracy_score(y_val, xgb_val_pred)
            xgb_val_logloss = log_loss(y_val, xgb_val_proba)

            print(f"\n=== Validation Set Performance ===")
            print(f"Random Forest - Val Accuracy: {rf_val_acc:.4f}, Val Log Loss: {rf_val_logloss:.4f}")
            print(f"XGBoost - Val Accuracy: {xgb_val_acc:.4f}, Val Log Loss: {xgb_val_logloss:.4f}")

            # Choose best model based on validation performance
            if rf_val_logloss < xgb_val_logloss:
                model = rf_model
                best_model_name = "RandomForest"
                val_acc = rf_val_acc
                val_logloss = rf_val_logloss
                cv_acc_mean = rf_cv_scores['test_accuracy'].mean()
                cv_logloss_mean = -rf_cv_scores['test_neg_log_loss'].mean()
            else:
                model = xgb_model
                best_model_name = "XGBoost"
                val_acc = xgb_val_acc
                val_logloss = xgb_val_logloss
                cv_acc_mean = xgb_cv_scores['test_accuracy'].mean()
                cv_logloss_mean = -xgb_cv_scores['test_neg_log_loss'].mean()

            print(f"\nSelected {best_model_name} as the best model")

            # Log both models and metrics
            mlflow.sklearn.log_model(rf_model, "random_forest_model")
            mlflow.sklearn.log_model(xgb_model, "xgboost_model")
            mlflow.sklearn.log_model(model, "best_model")

            # Log proper validation metrics (NOT training metrics)
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.log_metric("val_logloss", val_logloss)
            mlflow.log_metric("cv_accuracy_mean", cv_acc_mean)
            mlflow.log_metric("cv_logloss_mean", cv_logloss_mean)
            mlflow.log_metric("cv_accuracy_std", rf_cv_scores['test_accuracy'].std() if best_model_name == "RandomForest" else xgb_cv_scores['test_accuracy'].std())

            print(f"Final model validation metrics - Accuracy: {val_acc:.4f}, Log Loss: {val_logloss:.4f}")
            print(f"Cross-validation metrics - Accuracy: {cv_acc_mean:.4f}, Log Loss: {cv_logloss_mean:.4f}")
        else:
            # For larger datasets, continue with stacking approach
            print("Training stacked ensemble model with proper validation...")

            # Train XGBoost with early stopping on validation set
            xgb_model = XGBClassifier(
                use_label_encoder=False,
                eval_metric=cfg.get("training", {}).get("eval_metric", "logloss"),
                **cfg.get("xgboost", {}).get("hyperparameters", {})
            )
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=cfg.get("training", {}).get("early_stopping_rounds", 10),
                verbose=True
            )

            # Train stacked model on training data only
            model.fit(X_train, y_train)

            # Evaluate stacked model on validation set
            val_pred = model.predict(X_val)
            val_proba = model.predict_proba(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            val_logloss = log_loss(y_val, val_proba)

            # Also evaluate individual XGBoost model
            xgb_val_pred = xgb_model.predict(X_val)
            xgb_val_proba = xgb_model.predict_proba(X_val)
            xgb_val_acc = accuracy_score(y_val, xgb_val_pred)
            xgb_val_logloss = log_loss(y_val, xgb_val_proba)

            print(f"\n=== Validation Set Performance ===")
            print(f"Stacked Model - Val Accuracy: {val_acc:.4f}, Val Log Loss: {val_logloss:.4f}")
            print(f"XGBoost Model - Val Accuracy: {xgb_val_acc:.4f}, Val Log Loss: {xgb_val_logloss:.4f}")

            # Perform cross-validation on the full dataset for additional validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            cv_logloss_scores = cross_val_score(model, X, y, cv=5, scoring='neg_log_loss')

            print(f"Cross-validation - Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Cross-validation - Log Loss: {-cv_logloss_scores.mean():.4f} (+/- {cv_logloss_scores.std() * 2:.4f})")

            # Log models and proper validation metrics
            mlflow.sklearn.log_model(model, "stacked_model")
            mlflow.sklearn.log_model(xgb_model, "xgboost_model")

            # Log validation metrics (NOT training metrics)
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.log_metric("val_logloss", val_logloss)
            mlflow.log_metric("xgb_val_accuracy", xgb_val_acc)
            mlflow.log_metric("xgb_val_logloss", xgb_val_logloss)
            mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
            mlflow.log_metric("cv_accuracy_std", cv_scores.std())
            mlflow.log_metric("cv_logloss_mean", -cv_logloss_scores.mean())
            mlflow.log_metric("cv_logloss_std", cv_logloss_scores.std())

            # Print classification report for detailed analysis
            print(f"\n=== Detailed Classification Report (Validation Set) ===")
            print(classification_report(y_val, val_pred))
            print(f"\n=== Confusion Matrix (Validation Set) ===")
            print(confusion_matrix(y_val, val_pred))

        # Skip neural network training for small datasets
        if X.shape[0] <= 30:  # Updated threshold
            print("Skipping neural network training due to insufficient data")
            return  # Skip the rest of the training
        
        # Only run neural network training for larger datasets
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Get training config with defaults
        training_config = cfg.get("training", {})
        batch_size = min(training_config.get("batch_size", 1024), X.shape[0])  # Ensure batch size isn't larger than dataset
        early_stopping_rounds = training_config.get("early_stopping_rounds", 10)
        num_epochs = training_config.get("num_epochs", 20)
        learning_rate = training_config.get("lr", 1e-3)
        
        dataset = FootballDataset(df, use_focal)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        net = SimpleMLP(input_dim=X.shape[1]).to(device)
        optimizer = Adam(net.parameters(), lr=learning_rate)
        
        # Get focal loss parameters
        focal_params = {}
        if use_focal:
            class_balance_config = full_cfg.get("class_balance", {})
            focal_params = {
                "alpha": class_balance_config.get("alpha", 0.25),
                "gamma": class_balance_config.get("gamma", 2.0)
            }
            criterion = FocalLoss(**focal_params)
        else:
            criterion = nn.CrossEntropyLoss()

        trainer = create_supervised_trainer(net, optimizer, criterion, device=device)
        evaluator = create_supervised_evaluator(net, metrics={"loss": Loss(criterion)}, device=device)

        # Early stopping
        es = EarlyStopping(
            patience=early_stopping_rounds,
            score_function=lambda e: -e.state.metrics['loss'],
            trainer=trainer
        )
        evaluator.add_event_handler(Events.COMPLETED, es)

        # Model checkpoint
        chkpt = ModelCheckpoint(
            dirname="./models/checkpoints",
            filename_prefix="mlp",
            n_saved=1,
            create_dir=True,
            require_empty=False
        )
        evaluator.add_event_handler(Events.COMPLETED, chkpt, {"model": net})

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_epoch(engine):
            evaluator.run(loader)
            mlflow.log_metric("nn_loss", engine.state.output, step=engine.state.epoch)
            print(f"Epoch {engine.state.epoch}/{num_epochs}: loss={engine.state.output:.4f}")

        print(f"Starting neural network training for {num_epochs} epochs")
        trainer.run(loader, max_epochs=num_epochs)
        print("Neural network training complete")

        # Log PyTorch model
        mlflow.pytorch.log_model(net, "pytorch_model")
        
        # Evaluate on validation set (NOT training set) for final metrics
        try:
            # Use validation set for final evaluation
            if 'X_val' in locals() and 'y_val' in locals():
                val_preds = model.predict_proba(X_val)
                val_logloss = log_loss(y_val, val_preds)
                val_acc = accuracy_score(y_val, model.predict(X_val))

                mlflow.log_metric("final_val_logloss", val_logloss)
                mlflow.log_metric("final_val_acc", val_acc)
                print(f"Final validation metrics - Log Loss: {val_logloss:.4f}, Accuracy: {val_acc:.4f}")

                # Also log training metrics for comparison (to detect overfitting)
                train_preds = model.predict_proba(X_train)
                train_logloss = log_loss(y_train, train_preds)
                train_acc = accuracy_score(y_train, model.predict(X_train))

                mlflow.log_metric("train_logloss", train_logloss)
                mlflow.log_metric("train_acc", train_acc)

                # Calculate overfitting metrics
                acc_gap = train_acc - val_acc
                logloss_gap = val_logloss - train_logloss

                mlflow.log_metric("overfitting_acc_gap", acc_gap)
                mlflow.log_metric("overfitting_logloss_gap", logloss_gap)

                print(f"Training metrics - Log Loss: {train_logloss:.4f}, Accuracy: {train_acc:.4f}")
                print(f"Overfitting indicators - Acc Gap: {acc_gap:.4f}, LogLoss Gap: {logloss_gap:.4f}")

                if acc_gap > 0.1 or logloss_gap > 0.2:
                    print("⚠️  WARNING: Potential overfitting detected! Consider:")
                    print("   - Reducing model complexity")
                    print("   - Adding regularization")
                    print("   - Collecting more training data")
            else:
                print("Validation set not available for final evaluation")
        except Exception as e:
            print(f"Error evaluating model: {e}")
    def _normalize_features(self, features, params=None):
        """
        Normalize features with improved error handling and logging
        """
        self.logger.info("Normalizing features")
        normalized = features.copy()
        
        if params is None:
            # Load normalization parameters from config
            params = self.config.get('normalization_params', {})
        
        # Normalize each feature according to its parameters
        for col in normalized.columns:
            if col in params:
                # Check feature data type before normalization
                if pd.api.types.is_numeric_dtype(normalized[col]):
                    try:
                        mean = params[col].get('mean', 0)
                        std = params[col].get('std', 1)
                        if std > 0:  # Avoid division by zero
                            normalized[col] = (normalized[col] - mean) / std
                            self.logger.debug(f"Normalized {col} with mean={mean}, std={std}")
                        else:
                            self.logger.warning(f"Skipping normalization for {col}: std={std}")
                    except Exception as e:
                        self.logger.error(f"Error normalizing {col}: {e}")
                else:
                    self.logger.info(f"Skipping normalization for non-numeric feature: {col}")
        
        return normalized

    print("Training complete.")
