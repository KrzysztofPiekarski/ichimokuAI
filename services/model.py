import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from core.ichimoku import IchimokuFeatureExtractor
import logging
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
import joblib
from pathlib import Path


# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fallback implementations for missing modules
class IchimokuFeatureExtractor:
    """Fallback implementation of Ichimoku Feature Extractor"""
    
    def calculate_ichimoku(self, df: pd.DataFrame, tenkan_period: int = 9, 
                          kijun_period: int = 26, senkou_b_period: int = 52) -> pd.DataFrame:
        """Calculate Ichimoku Cloud indicators"""
        df = df.copy()
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = df['High'].rolling(window=tenkan_period).max()
        tenkan_low = df['Low'].rolling(window=tenkan_period).min()
        df['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = df['High'].rolling(window=kijun_period).max()
        kijun_low = df['Low'].rolling(window=kijun_period).min()
        df['kijun_sen'] = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(kijun_period)
        
        # Senkou Span B (Leading Span B)
        senkou_b_high = df['High'].rolling(window=senkou_b_period).max()
        senkou_b_low = df['Low'].rolling(window=senkou_b_period).min()
        df['senkou_span_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period)
        
        # Chikou Span (Lagging Span)
        df['chikou_span'] = df['Close'].shift(-kijun_period)
        
        return df

class FeatureAdder:
    """Fallback implementation of Feature Adder"""
    
    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI indicator"""
        df = df.copy()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df
    
    def add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicator"""
        df = df.copy()
        exp1 = df['Close'].ewm(span=fast).mean()
        exp2 = df['Close'].ewm(span=slow).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df

class IchimokuSignalGenerator:
    """Fallback implementation of Signal Generator"""
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Ichimoku signals"""
        return df


class IchimokuModel:
    """
    Machine Learning model for Ichimoku-based trading signals.
    
    Supports multiple ML algorithms and provides comprehensive feature engineering
    based on Ichimoku Cloud indicators and additional technical analysis.
    """
    
    SUPPORTED_MODELS = {
        'logistic': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier
    }
    
    SUPPORTED_SCALERS = {
        'standard': StandardScaler,
        'robust': RobustScaler
    }
    
    def __init__(self, model_type: str = 'logistic', scaler_type: str = 'standard',
                 use_class_weights: bool = True, random_state: int = 42):
        """
        Initialize Ichimoku ML Model
        
        Args:
            model_type: 'logistic', 'random_forest', or 'gradient_boosting'
            scaler_type: 'standard', 'robust', or None
            use_class_weights: Whether to balance class weights automatically
            random_state: Random state for reproducibility
        """
        # Initialize feature generators
        try:
            from core.ichimoku import IchimokuFeatureExtractor
            from core.features import FeatureAdder
            from core.signals import IchimokuSignalGenerator
            self.feature_extractor = IchimokuFeatureExtractor()
            self.feature_adder = FeatureAdder()
            self.signal_generator = IchimokuSignalGenerator()
            logger.info("✅ Załadowano moduły z core")
        except ImportError:
            logger.warning("⚠️ Używam implementacji fallback")
            self.feature_extractor = IchimokuFeatureExtractor()
            self.feature_adder = FeatureAdder()
            self.signal_generator = IchimokuSignalGenerator()
        
        # Model configuration
        self.model_type = model_type
        self.use_class_weights = use_class_weights
        self.random_state = random_state
        
        # Initialize model
        self.model = self._initialize_model(model_type)
        
        # Initialize scaler
        self.scaler = self._initialize_scaler(scaler_type)
        
        # Training state
        self.is_trained = False
        self.feature_columns = None
        self.model_metrics = {}
        self.training_config = {}

    def _initialize_model(self, model_type: str):
        """Initialize the ML model based on type"""
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"❌ Nieobsługiwany typ modelu: {model_type}. "
                           f"Dostępne: {list(self.SUPPORTED_MODELS.keys())}")
        
        model_class = self.SUPPORTED_MODELS[model_type]
        
        # Model-specific parameters
        if model_type == 'random_forest':
            params = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_state,
                'n_jobs': -1
            }
        elif model_type == 'gradient_boosting':
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': self.random_state
            }
        else:  # logistic
            params = {
                'random_state': self.random_state,
                'max_iter': 2000,
                'solver': 'liblinear'
            }
        
        # Add class weights if requested
        if self.use_class_weights and model_type != 'gradient_boosting':
            params['class_weight'] = 'balanced'
        
        return model_class(**params)

    def _initialize_scaler(self, scaler_type: Optional[str]):
        """Initialize the feature scaler"""
        if scaler_type is None:
            return None
        
        if scaler_type not in self.SUPPORTED_SCALERS:
            raise ValueError(f"❌ Nieobsługiwany typ scalera: {scaler_type}. "
                           f"Dostępne: {list(self.SUPPORTED_SCALERS.keys())}")
        
        return self.SUPPORTED_SCALERS[scaler_type]()

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate input dataframe structure"""
        if df is None or df.empty:
            raise ValueError("❌ DataFrame jest pusty lub None")
        
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"❌ Brakuje wymaganych kolumn: {missing_columns}")
        
        if len(df) < 100:  # Increased minimum for better model training
            raise ValueError(f"❌ Zbyt mało danych. Potrzeba co najmniej 100 wierszy, otrzymano: {len(df)}")
        
        # Check for reasonable price data
        for col in required_columns:
            if df[col].isnull().all():
                raise ValueError(f"❌ Kolumna {col} zawiera tylko wartości NaN")
            
            if (df[col] <= 0).any():
                logger.warning(f"⚠️ Kolumna {col} zawiera wartości <= 0")

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive feature set from OHLC data"""
        try:
            self.validate_dataframe(df)
            
            # Work on copy
            df_work = df.copy()
            
            # Ensure numeric data types
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'senkou_span_a', 'senkou_span_b']
            for col in numeric_columns:
                df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
            
            # Calculate Ichimoku indicators
            df_work = self.feature_extractor.calculate_ichimoku(df_work)
            
            # Validate Ichimoku columns
            expected_cols = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
            missing_cols = [col for col in expected_cols if col not in df_work.columns]
            if missing_cols:
                raise ValueError(f"❌ Brakuje kolumn Ichimoku: {missing_cols}")
            
            # Add technical indicators
            df_work = self.feature_adder.add_rsi(df_work)
            df_work = self.feature_adder.add_macd(df_work)
            
            # Create additional features
            df_work = self._create_additional_features(df_work)
            
            # Define feature columns
            feature_columns = [
                'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 
                'chikou_span', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'cloud_thickness', 'price_vs_cloud', 'tenkan_kijun_cross', 
                'price_momentum', 'volatility', 'price_position', 'cloud_twist'
            ]
            
            # Filter available features
            available_features = [col for col in feature_columns if col in df_work.columns]
            
            if not available_features:
                raise ValueError("❌ Nie udało się wygenerować żadnych cech")
            
            features_df = df_work[available_features].copy()
            
            # Store feature columns
            if not self.is_trained:  # Only update during training
                self.feature_columns = available_features
            
            logger.info(f"✅ Wygenerowano {len(available_features)} cech")
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas generowania cech: {e}")
            raise

    def _create_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive derived features"""
        df = df.copy()
        
        try:
            # Cloud thickness (normalized by price)
            if all(col in df.columns for col in ['senkou_span_a', 'senkou_span_b', 'Close']):
                df['cloud_thickness'] = abs(df['senkou_span_a'] - df['senkou_span_b']) / df['Close']
            
            # Price position relative to cloud (enhanced)
            if all(col in df.columns for col in ['Close', 'senkou_span_a', 'senkou_span_b']):
                cloud_top = np.maximum(df['senkou_span_a'], df['senkou_span_b'])
                cloud_bottom = np.minimum(df['senkou_span_a'], df['senkou_span_b'])
                
                # Detailed position
                df['price_vs_cloud'] = np.where(
                    df['Close'] > cloud_top, 1,  # Above cloud
                    np.where(df['Close'] < cloud_bottom, -1, 0)  # Below cloud or inside
                )
                
                # Distance from cloud (normalized)
                cloud_middle = (cloud_top + cloud_bottom) / 2
                df['price_position'] = (df['Close'] - cloud_middle) / df['Close']
            
            # Tenkan-Kijun relationship
            if all(col in df.columns for col in ['tenkan_sen', 'kijun_sen']):
                df['tenkan_kijun_cross'] = np.where(df['tenkan_sen'] > df['kijun_sen'], 1, -1)
                df['tenkan_kijun_diff'] = (df['tenkan_sen'] - df['kijun_sen']) / df['kijun_sen']
            
            # Price momentum (multiple timeframes)
            if 'Close' in df.columns:
                df['price_momentum'] = df['Close'].pct_change(periods=5)
                df['price_momentum_long'] = df['Close'].pct_change(periods=20)
            
            # Volatility
            if 'High' in df.columns and 'Low' in df.columns:
                df['volatility'] = (df['High'] - df['Low']) / df['Close']
            
            # Cloud twist (cloud direction change)
            if all(col in df.columns for col in ['senkou_span_a', 'senkou_span_b']):
                span_a_trend = df['senkou_span_a'].diff()
                span_b_trend = df['senkou_span_b'].diff()
                df['cloud_twist'] = np.where(
                    (span_a_trend > 0) & (span_b_trend > 0), 1,
                    np.where((span_a_trend < 0) & (span_b_trend < 0), -1, 0)
                )
            
            # RSI divergence (if RSI exists)
            if 'rsi' in df.columns:
                df['rsi_oversold'] = np.where(df['rsi'] < 30, 1, 0)
                df['rsi_overbought'] = np.where(df['rsi'] > 70, 1, 0)
            
        except Exception as e:
            logger.warning(f"⚠️ Błąd podczas tworzenia dodatkowych cech: {e}")
        
        return df

    def generate_target_signals(self, df: pd.DataFrame, 
                              future_periods: int = 5,
                              return_threshold: float = 0.002) -> pd.DataFrame:
        """Generate sophisticated target signals"""
        df = df.copy()
        
        try:
            required_columns = ['Close', 'senkou_span_a', 'senkou_span_b']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"❌ Brak wymaganych kolumn: {missing_cols}")
            
            # Cloud-based signals
            cloud_top = np.maximum(df['senkou_span_a'], df['senkou_span_b'])
            cloud_bottom = np.minimum(df['senkou_span_a'], df['senkou_span_b'])
            
            # Basic signals
            df['bull_signal'] = np.where(df['Close'] > cloud_top, 1, 0)
            df['bear_signal'] = np.where(df['Close'] < cloud_bottom, 1, 0)
            df['combined_signal'] = np.where(
                df['bull_signal'] == 1, 1,
                np.where(df['bear_signal'] == 1, -1, 0)
            )
            
            # Future return-based target (more sophisticated)
            df['future_return'] = df['Close'].shift(-future_periods) / df['Close'] - 1
            
            # Multi-class target with different thresholds
            df['target_signal'] = np.where(
                df['future_return'] > return_threshold, 1,  # Strong buy
                np.where(df['future_return'] < -return_threshold, 0, 2)  # Strong sell vs Hold
            )
            
            # Binary target for simpler classification
            df['target_binary'] = np.where(df['future_return'] > return_threshold, 1, 0)
            
            # Trend-following target
            price_trend = df['Close'].rolling(window=10).mean().diff()
            df['trend_signal'] = np.where(price_trend > 0, 1, 0)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas generowania sygnałów: {e}")
            raise

    def train_model(self, df: pd.DataFrame, target_type: str = 'target_binary',
                    test_size: float = 0.2, cv_folds: int = 5,
                    use_time_series_split: bool = True) -> Dict[str, Any]:
        """
        Train the ML model with comprehensive evaluation
        
        Args:
            df: Input dataframe with OHLC data
            target_type: Type of target signal
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            use_time_series_split: Use time-aware cross-validation
        """
        try:
            logger.info("🚀 Rozpoczynam trenowanie modelu...")
            
            # Generate features and targets
            features_df = self.generate_features(df)
            signals_df = self.generate_target_signals(df)
            
            # Store training configuration
            self.training_config = {
                'target_type': target_type,
                'model_type': self.model_type,
                'scaler_type': type(self.scaler).__name__ if self.scaler else None,
                'use_class_weights': self.use_class_weights
            }
            
            # Align data
            common_index = features_df.index.intersection(signals_df.index)
            X = features_df.loc[common_index]
            y = signals_df.loc[common_index, target_type]
            
            # Clean data
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                raise ValueError("❌ Brak danych po oczyszczeniu")
            
            logger.info(f"📊 Dane treningowe: {len(X)} próbek, {len(X.columns)} cech")
            
            # Analyze class distribution
            class_distribution = y.value_counts().sort_index()
            logger.info(f"📈 Rozkład klas: {dict(class_distribution)}")
            
            # Check for class imbalance
            min_class_size = class_distribution.min()
            max_class_size = class_distribution.max()
            imbalance_ratio = max_class_size / min_class_size if min_class_size > 0 else float('inf')
            
            if imbalance_ratio > 10:
                logger.warning(f"⚠️ Wykryto niezbalansowane klasy (ratio: {imbalance_ratio:.1f})")
            
            # Split data (time-aware for financial data)
            if use_time_series_split:
                split_point = int(len(X) * (1 - test_size))
                X_train = X.iloc[:split_point]
                X_test = X.iloc[split_point:]
                y_train = y.iloc[:split_point]
                y_test = y.iloc[split_point:]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state,
                    stratify=y if len(y.unique()) > 1 else None
                )
            
            # Feature scaling
            if self.scaler:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_train_scaled = X_train.values
                X_test_scaled = X_test.values
            
            # Train model
            logger.info(f"⚙️ Trenowanie modelu {self.model_type}...")
            self.model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)
            
            # Comprehensive evaluation
            metrics = self._calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
            
            # Cross-validation
            if use_time_series_split:
                cv = TimeSeriesSplit(n_splits=cv_folds)
            else:
                cv = cv_folds
            
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            
            # Final metrics
            self.model_metrics = {
                **metrics,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'class_distribution': dict(class_distribution),
                'imbalance_ratio': imbalance_ratio
            }
            
            self.is_trained = True
            
            # Log results
            logger.info(f"✅ Model wytrenowany!")
            logger.info(f"📊 Dokładność: {metrics['accuracy']:.3f}")
            logger.info(f"📊 Precision: {metrics['precision']:.3f}")
            logger.info(f"📊 Recall: {metrics['recall']:.3f}")
            logger.info(f"📊 F1-Score: {metrics['f1_score']:.3f}")
            logger.info(f"📊 CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            return self.model_metrics
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas trenowania modelu: {e}")
            raise

    def _calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive model evaluation metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 (weighted average for multiclass)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True
        )
        
        # Prediction confidence
        if y_pred_proba is not None:
            metrics['mean_confidence'] = np.max(y_pred_proba, axis=1).mean()
            metrics['min_confidence'] = np.max(y_pred_proba, axis=1).min()
            metrics['max_confidence'] = np.max(y_pred_proba, axis=1).max()
        
        return metrics

    def predict_signal(self, df: pd.DataFrame, return_confidence: bool = True) -> Dict[str, Any]:
        """
        Enhanced prediction with confidence scores and risk assessment
        """
        try:
            if not self.is_trained:
                raise ValueError("❌ Model nie został wytrenowany")
            
            # Generate features
            features_df = self.generate_features(df)
            
            # Use training features
            if self.feature_columns:
                missing_features = [col for col in self.feature_columns if col not in features_df.columns]
                if missing_features:
                    logger.warning(f"⚠️ Brakuje cech: {missing_features}")
                    available_features = [col for col in self.feature_columns if col in features_df.columns]
                    if not available_features:
                        raise ValueError("❌ Brak dostępnych cech do przewidywania")
                    X = features_df[available_features]
                else:
                    X = features_df[self.feature_columns]
            else:
                X = features_df
            
            # Clean data
            X_clean = X.dropna()
            original_indices = X_clean.index
            
            if len(X_clean) == 0:
                raise ValueError("❌ Brak prawidłowych danych do przewidywania")
            
            # Scale features
            if self.scaler:
                X_scaled = self.scaler.transform(X_clean)
            else:
                X_scaled = X_clean.values
            
            # Predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            # Calculate confidence and risk metrics
            max_proba = np.max(probabilities, axis=1)
            prediction_confidence = max_proba
            
            # Risk assessment based on confidence
            risk_level = np.where(
                max_proba > 0.8, 'Low',
                np.where(max_proba > 0.6, 'Medium', 'High')
            )
            
            result = {
                'predictions': predictions.tolist(),
                'indices': original_indices.tolist(),
                'signal_strength': prediction_confidence.tolist(),
                'risk_level': risk_level.tolist(),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            if return_confidence:
                result['probabilities'] = probabilities.tolist()
                result['confidence_stats'] = {
                    'mean': float(prediction_confidence.mean()),
                    'std': float(prediction_confidence.std()),
                    'min': float(prediction_confidence.min()),
                    'max': float(prediction_confidence.max())
                }
            
            logger.info(f"🎯 Przewidywanie: {len(predictions)} sygnałów, "
                       f"średnia pewność: {prediction_confidence.mean():.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas przewidywania: {e}")
            raise

    def save_model(self, filepath: Union[str, Path]) -> None:
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("❌ Model nie został wytrenowany")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_metrics': self.model_metrics,
            'training_config': self.training_config,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"✅ Model zapisany: {filepath}")

    def load_model(self, filepath: Union[str, Path]) -> None:
        """Load trained model from file"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"❌ Plik modelu nie istnieje: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_metrics = model_data['model_metrics']
        self.training_config = model_data.get('training_config', {})
        self.model_type = model_data.get('model_type', 'unknown')
        self.is_trained = True
        
        logger.info(f"✅ Model załadowany: {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        if not self.is_trained:
            return {"status": "Model nie został wytrenowany"}
        
        info = {
            "model_type": self.model_type,
            "model_class": type(self.model).__name__,
            "features_used": self.feature_columns,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "scaler_used": type(self.scaler).__name__ if self.scaler else None,
            "training_config": self.training_config,
            "metrics": self.model_metrics,
            "is_trained": self.is_trained
        }
        
        return info

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """Get feature importance with ranking"""
        if not self.is_trained:
            raise ValueError("❌ Model nie został wytrenowany")
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("⚠️ Model nie posiada informacji o ważności cech")
            return {}
        
        if not self.feature_columns:
            logger.warning("⚠️ Brak informacji o kolumnach cech")
            return {}
        
        try:
            # Get feature importances
            importances = self.model.feature_importances_
            
            # Create feature importance dictionary
            feature_importance = dict(zip(self.feature_columns, importances))
            
            # Sort by importance (descending)
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Return top N features
            top_features = dict(sorted_features[:top_n])
            
            logger.info(f"📊 Top {len(top_features)} najważniejszych cech:")
            for feature, importance in top_features.items():
                logger.info(f"  {feature}: {importance:.4f}")
            
            return top_features
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas pobierania ważności cech: {e}")
            return {}

    def optimize_hyperparameters(self, df: pd.DataFrame, target_type: str = 'target_binary',
                                cv_folds: int = 3, n_iter: int = 20) -> Dict[str, Any]:
        """
        Optimize model hyperparameters using RandomizedSearchCV
        
        Args:
            df: Input dataframe with OHLC data
            target_type: Type of target signal
            cv_folds: Number of cross-validation folds
            n_iter: Number of parameter combinations to try
        """
        try:
            from sklearn.model_selection import RandomizedSearchCV
            from scipy.stats import randint, uniform
            
            logger.info("🔧 Rozpoczynam optymalizację hiperparametrów...")
            
            # Generate features and targets
            features_df = self.generate_features(df)
            signals_df = self.generate_target_signals(df)
            
            # Align data
            common_index = features_df.index.intersection(signals_df.index)
            X = features_df.loc[common_index]
            y = signals_df.loc[common_index, target_type]
            
            # Clean data
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                raise ValueError("❌ Brak danych po oczyszczeniu")
            
            # Scale features
            if self.scaler:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X.values
            
            # Define parameter grids for different models
            param_grids = {
                'random_forest': {
                    'n_estimators': randint(50, 500),
                    'max_depth': randint(3, 20),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'max_features': ['sqrt', 'log2', None]
                },
                'gradient_boosting': {
                    'n_estimators': randint(50, 300),
                    'learning_rate': uniform(0.01, 0.3),
                    'max_depth': randint(3, 10),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10)
                },
                'logistic': {
                    'C': uniform(0.1, 10),
                    'solver': ['liblinear', 'lbfgs'],
                    'max_iter': randint(1000, 3000)
                }
            }
            
            param_grid = param_grids.get(self.model_type, {})
            
            if not param_grid:
                logger.warning(f"⚠️ Brak zdefiniowanej siatki parametrów dla {self.model_type}")
                return {"error": "Brak zdefiniowanej siatki parametrów"}
            
            # Create base model
            base_model = self._initialize_model(self.model_type)
            
            # Randomized search
            random_search = RandomizedSearchCV(
                base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring='accuracy',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
            
            # Fit the search
            random_search.fit(X_scaled, y)
            
            # Update model with best parameters
            self.model = random_search.best_estimator_
            
            # Get optimization results
            optimization_results = {
                'best_params': random_search.best_params_,
                'best_score': random_search.best_score_,
                'best_score_std': random_search.cv_results_['std_test_score'][random_search.best_index_],
                'n_iterations': n_iter,
                'cv_folds': cv_folds
            }
            
            logger.info(f"✅ Optymalizacja zakończona!")
            logger.info(f"📊 Najlepszy wynik: {random_search.best_score_:.4f}")
            logger.info(f"📊 Najlepsze parametry: {random_search.best_params_}")
            
            return optimization_results
            
        except ImportError:
            logger.error("❌ Brak wymaganych bibliotek do optymalizacji")
            return {"error": "Brak wymaganych bibliotek"}
        except Exception as e:
            logger.error(f"❌ Błąd podczas optymalizacji: {e}")
            raise

    def evaluate_model_stability(self, df: pd.DataFrame, target_type: str = 'target_binary',
                               n_splits: int = 5, test_periods: int = 3) -> Dict[str, Any]:
        """
        Evaluate model stability across different time periods
        
        Args:
            df: Input dataframe with OHLC data
            target_type: Type of target signal
            n_splits: Number of time-based splits
            test_periods: Number of recent periods to test
        """
        try:
            logger.info("📈 Rozpoczynam analizę stabilności modelu...")
            
            # Generate features and targets
            features_df = self.generate_features(df)
            signals_df = self.generate_target_signals(df)
            
            # Align data
            common_index = features_df.index.intersection(signals_df.index)
            X = features_df.loc[common_index]
            y = signals_df.loc[common_index, target_type]
            
            # Clean data
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                raise ValueError("❌ Brak danych po oczyszczeniu")
            
            # Time-based splits for stability testing
            data_length = len(X)
            split_size = data_length // n_splits
            
            stability_scores = []
            period_results = []
            
            for i in range(test_periods):
                # Define training and test periods
                test_end = data_length - i * split_size
                test_start = max(0, test_end - split_size)
                train_end = test_start
                train_start = max(0, train_end - split_size * 2)  # Use 2x data for training
                
                if train_start >= train_end or test_start >= test_end:
                    continue
                
                # Split data
                X_train_period = X.iloc[train_start:train_end]
                y_train_period = y.iloc[train_start:train_end]
                X_test_period = X.iloc[test_start:test_end]
                y_test_period = y.iloc[test_start:test_end]
                
                if len(X_train_period) == 0 or len(X_test_period) == 0:
                    continue
                
                # Scale features
                if self.scaler:
                    period_scaler = type(self.scaler)()
                    X_train_scaled = period_scaler.fit_transform(X_train_period)
                    X_test_scaled = period_scaler.transform(X_test_period)
                else:
                    X_train_scaled = X_train_period.values
                    X_test_scaled = X_test_period.values
                
                # Train period-specific model
                period_model = self._initialize_model(self.model_type)
                period_model.fit(X_train_scaled, y_train_period)
                
                # Evaluate
                y_pred = period_model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test_period, y_pred)
                
                stability_scores.append(accuracy)
                period_results.append({
                    'period': i + 1,
                    'train_samples': len(X_train_period),
                    'test_samples': len(X_test_period),
                    'accuracy': accuracy,
                    'train_start_idx': train_start,
                    'test_end_idx': test_end
                })
                
                logger.info(f"📊 Okres {i+1}: Dokładność = {accuracy:.4f}")
            
            if not stability_scores:
                return {"error": "Nie udało się przeprowadzić analizy stabilności"}
            
            # Calculate stability metrics
            stability_results = {
                'period_scores': stability_scores,
                'period_details': period_results,
                'mean_accuracy': np.mean(stability_scores),
                'std_accuracy': np.std(stability_scores),
                'min_accuracy': np.min(stability_scores),
                'max_accuracy': np.max(stability_scores),
                'stability_coefficient': 1 - (np.std(stability_scores) / np.mean(stability_scores)) if np.mean(stability_scores) > 0 else 0,
                'n_periods_tested': len(stability_scores)
            }
            
            # Stability assessment
            stability_coeff = stability_results['stability_coefficient']
            if stability_coeff > 0.9:
                stability_assessment = "Bardzo stabilny"
            elif stability_coeff > 0.8:
                stability_assessment = "Stabilny"
            elif stability_coeff > 0.7:
                stability_assessment = "Umiarkowanie stabilny"
            else:
                stability_assessment = "Niestabilny"
            
            stability_results['stability_assessment'] = stability_assessment
            
            logger.info(f"✅ Analiza stabilności zakończona!")
            logger.info(f"📊 Średnia dokładność: {stability_results['mean_accuracy']:.4f} ± {stability_results['std_accuracy']:.4f}")
            logger.info(f"📊 Ocena stabilności: {stability_assessment}")
            
            return stability_results
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas analizy stabilności: {e}")
            raise

    def generate_trading_report(self, df: pd.DataFrame, 
                              include_backtest: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive trading report with model insights
        
        Args:
            df: Input dataframe with OHLC data
            include_backtest: Whether to include backtest results
        """
        try:
            if not self.is_trained:
                raise ValueError("❌ Model nie został wytrenowany")
            
            logger.info("📋 Generuję raport handlowy...")
            
            # Get predictions
            predictions_result = self.predict_signal(df, return_confidence=True)
            
            # Basic statistics
            predictions = np.array(predictions_result['predictions'])
            confidences = np.array(predictions_result['signal_strength'])
            
            # Signal distribution
            unique, counts = np.unique(predictions, return_counts=True)
            signal_distribution = dict(zip(unique.astype(int), counts.astype(int)))
            
            # Confidence analysis
            confidence_stats = {
                'mean_confidence': float(np.mean(confidences)),
                'median_confidence': float(np.median(confidences)),
                'high_confidence_signals': int(np.sum(confidences > 0.8)),
                'medium_confidence_signals': int(np.sum((confidences > 0.6) & (confidences <= 0.8))),
                'low_confidence_signals': int(np.sum(confidences <= 0.6))
            }
            
            # Recent signals (last 20)
            recent_count = min(20, len(predictions))
            recent_signals = {
                'count': recent_count,
                'signals': predictions[-recent_count:].tolist(),
                'confidences': confidences[-recent_count:].tolist(),
                'indices': predictions_result['indices'][-recent_count:]
            }
            
            # Model performance summary
            performance_summary = {
                'model_type': self.model_type,
                'training_accuracy': self.model_metrics.get('accuracy', 'N/A'),
                'cv_score': self.model_metrics.get('cv_mean', 'N/A'),
                'feature_count': len(self.feature_columns) if self.feature_columns else 0
            }
            
            # Feature importance (if available)
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = self.get_feature_importance(top_n=5)
            
            # Risk assessment
            risk_analysis = {
                'high_risk_predictions': int(np.sum(confidences < 0.6)),
                'risk_percentage': float((np.sum(confidences < 0.6) / len(confidences)) * 100),
                'average_risk_level': 'High' if np.mean(confidences) < 0.6 else 'Medium' if np.mean(confidences) < 0.8 else 'Low'
            }
            
            # Compile report
            report = {
                'generated_at': pd.Timestamp.now().isoformat(),
                'data_period': {
                    'start': str(df.index[0]) if not df.empty else 'N/A',
                    'end': str(df.index[-1]) if not df.empty else 'N/A',
                    'total_periods': len(df)
                },
                'model_info': performance_summary,
                'signal_analysis': {
                    'total_signals': len(predictions),
                    'signal_distribution': signal_distribution,
                    'confidence_analysis': confidence_stats,
                    'recent_signals': recent_signals
                },
                'risk_assessment': risk_analysis,
                'feature_importance': feature_importance,
                'recommendations': self._generate_recommendations(signal_distribution, confidence_stats, risk_analysis)
            }
            
            # Add backtest if requested
            if include_backtest and len(df) > 100:
                try:
                    backtest_results = self._simple_backtest(df, predictions_result)
                    report['backtest_results'] = backtest_results
                except Exception as e:
                    logger.warning(f"⚠️ Nie udało się przeprowadzić backtesta: {e}")
                    report['backtest_results'] = {"error": str(e)}
            
            logger.info("✅ Raport handlowy wygenerowany!")
            return report
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas generowania raportu: {e}")
            raise

    def _generate_recommendations(self, signal_dist: Dict, confidence_stats: Dict, 
                                risk_analysis: Dict) -> List[str]:
        """Generate trading recommendations based on analysis"""
        recommendations = []
        
        # Signal distribution analysis
        total_signals = sum(signal_dist.values())
        if total_signals > 0:
            buy_ratio = signal_dist.get(1, 0) / total_signals
            if buy_ratio > 0.7:
                recommendations.append("🔥 Silny sygnał zakupu - rozważ zwiększenie pozycji")
            elif buy_ratio < 0.3:
                recommendations.append("📉 Przeważają sygnały sprzedaży - rozważ zmniejszenie pozycji")
            else:
                recommendations.append("⚖️ Mieszane sygnały - zachowaj ostrożność")
        
        # Confidence analysis
        if confidence_stats['mean_confidence'] > 0.8:
            recommendations.append("✅ Wysokie zaufanie do sygnałów - można działać z większą pewnością")
        elif confidence_stats['mean_confidence'] < 0.6:
            recommendations.append("⚠️ Niskie zaufanie do sygnałów - zwiększ ostrożność")
        
        # Risk analysis
        if risk_analysis['risk_percentage'] > 50:
            recommendations.append("🚨 Wysoki poziom ryzyka - rozważ zmniejszenie wielkości pozycji")
        elif risk_analysis['risk_percentage'] < 20:
            recommendations.append("🛡️ Niski poziom ryzyka - można rozważyć większe pozycje")
        
        # General recommendations
        recommendations.append("📊 Regularnie monitoruj wydajność modelu")
        recommendations.append("🔄 Rozważ retrenowanie modelu przy znaczących zmianach rynkowych")
        
        return recommendations

    def _simple_backtest(self, df: pd.DataFrame, predictions_result: Dict) -> Dict[str, Any]:
        """Simple backtest implementation"""
        try:
            predictions = np.array(predictions_result['predictions'])
            indices = predictions_result['indices']
            
            # Align predictions with price data
            prices = df['Close'].loc[indices]
            
            # Simple strategy: buy on signal 1, sell on signal 0
            returns = []
            positions = []
            current_position = 0
            
            for i, (signal, price) in enumerate(zip(predictions, prices)):
                if i == 0:
                    positions.append(0)
                    returns.append(0)
                    continue
                
                prev_price = prices.iloc[i-1]
                price_return = (price - prev_price) / prev_price
                
                # Update position based on previous signal
                if i > 0:
                    prev_signal = predictions[i-1]
                    current_position = 1 if prev_signal == 1 else 0
                
                # Calculate strategy return
                strategy_return = current_position * price_return
                returns.append(strategy_return)
                positions.append(current_position)
            
            returns = np.array(returns)
            
            # Calculate metrics
            total_return = np.sum(returns)
            win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
            avg_return = np.mean(returns) if len(returns) > 0 else 0
            volatility = np.std(returns) if len(returns) > 1 else 0
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            
            backtest_results = {
                'total_return': float(total_return),
                'win_rate': float(win_rate),
                'average_return': float(avg_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'number_of_trades': int(np.sum(np.diff(positions) != 0)),
                'max_drawdown': float(np.min(np.cumsum(returns))) if len(returns) > 0 else 0
            }
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas backtesta: {e}")
            return {"error": str(e)}
