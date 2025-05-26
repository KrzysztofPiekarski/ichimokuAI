import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging

# Assuming these modules exist - if not, we'll need to implement them
try:
    from core.ichimoku import IchimokuFeatureExtractor
    from core.features import FeatureAdder
    from core.signals import IchimokuSignalGenerator
except ImportError:
    # Fallback implementations if modules don't exist
    class IchimokuFeatureExtractor:
        def calculate_ichimoku(self, df, tenkan_period=9, kijun_period=26, senkou_b_period=52):
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
        def add_rsi(self, df, period=14):
            """Add RSI indicator"""
            df = df.copy()
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            return df
        
        def add_macd(self, df, fast=12, slow=26, signal=9):
            """Add MACD indicator"""
            df = df.copy()
            exp1 = df['Close'].ewm(span=fast).mean()
            exp2 = df['Close'].ewm(span=slow).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=signal).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            return df
    
    class IchimokuSignalGenerator:
        def generate_signals(self, df):
            """Generate Ichimoku signals"""
            return df

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IchimokuModel:
    def __init__(self, model_type='logistic', use_scaling=True):
        """
        Initialize Ichimoku ML Model
        
        Args:
            model_type: 'logistic' or 'random_forest'
            use_scaling: Whether to scale features
        """
        self.feature_extractor = IchimokuFeatureExtractor()
        self.feature_adder = FeatureAdder()
        self.signal_generator = IchimokuSignalGenerator()
        
        # Model selection
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        
        self.scaler = StandardScaler() if use_scaling else None
        self.is_trained = False
        self.feature_columns = None
        self.model_metrics = {}

    def validate_dataframe(self, df):
        """Validate input dataframe structure"""
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"❌ Brakuje wymaganych kolumn: {missing_columns}")
        
        if df.empty:
            raise ValueError("❌ DataFrame jest pusty")
        
        if len(df) < 52:  # Need at least 52 periods for Ichimoku
            raise ValueError(f"❌ Zbyt mało danych. Potrzeba co najmniej 52 wierszy, otrzymano: {len(df)}")
    
    def generate_features(self, df):
        """
        Generate input features based on Ichimoku and technical indicators
        """
        try:
            self.validate_dataframe(df)
            
            # Make a copy to avoid modifying original data
            df_work = df.copy()
            
            # Ensure numeric data types
            numeric_columns = ['Open', 'High', 'Low', 'Close']
            for col in numeric_columns:
                df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
            
             # Calculate Ichimoku indicators
            df_work = self.feature_extractor.calculate_ichimoku(df_work)

            # ✅ Walidacja obecności kolumn Ichimoku
            expected_cols = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
            missing_cols = [col for col in expected_cols if col not in df_work.columns]
            if missing_cols:
                raise ValueError(f"❌ Brakuje kolumn Ichimoku: {missing_cols}")
                                 
            # Add additional technical indicators
            df_work = self.feature_adder.add_rsi(df_work)
            df_work = self.feature_adder.add_macd(df_work)
            
            # Create additional features
            df_work = self._create_additional_features(df_work)
            
            # Select feature columns
            feature_columns = [
                'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 
                'chikou_span', 'rsi', 'macd', 'cloud_thickness', 'price_vs_cloud',
                'tenkan_kijun_cross', 'price_momentum'
            ]
            
            # Filter only existing columns
            available_features = [col for col in feature_columns if col in df_work.columns]
            
            if not available_features:
                raise ValueError("❌ Nie udało się wygenerować żadnych cech")
            
            features_df = df_work[available_features].copy()
            
            # Store feature column names for later use
            self.feature_columns = available_features
            
            logger.info(f"✅ Wygenerowano {len(available_features)} cech: {available_features}")
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas generowania cech: {e}")
            raise

    def _create_additional_features(self, df):
        """Create additional derived features"""
        df = df.copy()
        
        try:
            # Cloud thickness
            if 'senkou_span_a' in df.columns and 'senkou_span_b' in df.columns:
                df['cloud_thickness'] = abs(df['senkou_span_a'] - df['senkou_span_b'])
            
            # Price position relative to cloud
            if all(col in df.columns for col in ['Close', 'senkou_span_a', 'senkou_span_b']):
                cloud_top = np.maximum(df['senkou_span_a'], df['senkou_span_b'])
                cloud_bottom = np.minimum(df['senkou_span_a'], df['senkou_span_b'])
                
                df['price_vs_cloud'] = np.where(
                    df['Close'] > cloud_top, 1,  # Above cloud
                    np.where(df['Close'] < cloud_bottom, -1, 0)  # Below cloud or inside
                )
            
            # Tenkan-Kijun cross
            if 'tenkan_sen' in df.columns and 'kijun_sen' in df.columns:
                df['tenkan_kijun_cross'] = np.where(
                    df['tenkan_sen'] > df['kijun_sen'], 1, -1
                )
            
            # Price momentum
            if 'Close' in df.columns:
                df['price_momentum'] = df['Close'].pct_change(periods=5)
            
        except Exception as e:
            logger.warning(f"⚠️ Błąd podczas tworzenia dodatkowych cech: {e}")
        
        return df

    def generate_target_signals(self, df):
        """
        Generate target signals for training
        """
        df = df.copy()
        
        try:
            # Ensure required columns exist
            required_columns = ['Close', 'senkou_span_a', 'senkou_span_b']
            for col in required_columns:
                if col not in df.columns:
                    raise KeyError(f"❌ Brak wymaganej kolumny: {col}")
            
            # Multiple signal strategies
            cloud_top = np.maximum(df['senkou_span_a'], df['senkou_span_b'])
            cloud_bottom = np.minimum(df['senkou_span_a'], df['senkou_span_b'])
            
            # Bull signal: Price above cloud
            df['bull_signal'] = np.where(df['Close'] > cloud_top, 1, 0)
            
            # Bear signal: Price below cloud
            df['bear_signal'] = np.where(df['Close'] < cloud_bottom, 1, 0)
            
            # Combined signal: Bull=1, Bear=-1, Neutral=0
            df['combined_signal'] = np.where(
                df['bull_signal'] == 1, 1,
                np.where(df['bear_signal'] == 1, -1, 0)
            )
            
            # Future price direction (for supervised learning)
            future_periods = 5
            df['future_return'] = df['Close'].shift(-future_periods) / df['Close'] - 1
            df['target_signal'] = np.where(df['future_return'] > 0.001, 1, 0)  # 0.1% threshold
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas generowania sygnałów: {e}")
            raise

    def train_model(self, df, target_type='bull_signal', test_size=0.2):
        """
        Train the ML model on Ichimoku data
        
        Args:
            df: Input dataframe with OHLC data
            target_type: Type of target signal ('bull_signal', 'bear_signal', 'target_signal')
            test_size: Proportion of data for testing
        """
        try:
            logger.info("🚀 Rozpoczynam trenowanie modelu...")
            
            # Generate features and signals
            features_df = self.generate_features(df)
            signals_df = self.generate_target_signals(df)
            
            # Align features and targets
            common_index = features_df.index.intersection(signals_df.index)
            X = features_df.loc[common_index]
            y = signals_df.loc[common_index, target_type]
            
            # Remove NaN values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                raise ValueError("❌ Brak danych po usunięciu NaN")
            
            logger.info(f"📊 Dane treningowe: {len(X)} próbek, {len(X.columns)} cech")
            
            # Check class distribution
            class_distribution = y.value_counts()
            logger.info(f"📈 Rozkład klas: {dict(class_distribution)}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale features if scaler is available
            if self.scaler:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Train model
            logger.info("⚙️ Trenowanie modelu...")
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            
            # Store metrics
            self.model_metrics = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            self.is_trained = True
            
            logger.info(f"✅ Model wytrenowany!")
            logger.info(f"📊 Dokładność: {accuracy:.3f}")
            logger.info(f"📊 CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            return self.model_metrics
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas trenowania modelu: {e}")
            raise

    def predict_signal(self, df):
        """
        Predict signal based on Ichimoku features
        
        Args:
            df: Input dataframe (can be single row or multiple rows)
            
        Returns:
            Array of predictions or single prediction
        """
        try:
            if not self.is_trained:
                raise ValueError("❌ Model nie został wytrenowany. Wywołaj najpierw train_model()")
            
            # Generate features
            features_df = self.generate_features(df)
            
            # Select only columns used during training
            if self.feature_columns:
                missing_features = [col for col in self.feature_columns if col not in features_df.columns]
                if missing_features:
                    raise ValueError(f"❌ Brakuje cech użytych podczas trenowania: {missing_features}")
                
                X = features_df[self.feature_columns]
            else:
                X = features_df
            
            # Remove NaN values
            X_clean = X.dropna()
            
            if len(X_clean) == 0:
                raise ValueError("❌ Brak prawidłowych danych do przewidywania")
            
            # Scale features if scaler was used during training
            if self.scaler:
                X_scaled = self.scaler.transform(X_clean)
            else:
                X_scaled = X_clean
            
            # Make prediction
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            # Return predictions with probabilities
            result = {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'signal_strength': np.max(probabilities, axis=1).tolist()
            }
            
            logger.info(f"🎯 Przewidywanie zakończone: {len(predictions)} sygnałów")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas przewidywania: {e}")
            raise

    def get_model_info(self):
        """Get information about the trained model"""
        if not self.is_trained:
            return {"status": "Model nie został wytrenowany"}
        
        info = {
            "model_type": type(self.model).__name__,
            "features_used": self.feature_columns,
            "is_scaled": self.scaler is not None,
            "metrics": self.model_metrics
        }
        
        return info

    def get_feature_importance(self):
        """Get feature importance (for supported models)"""
        if not self.is_trained:
            raise ValueError("❌ Model nie został wytrenowany")
        
        if hasattr(self.model, 'feature_importances_'):
            # Random Forest
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Logistic Regression
            importance = np.abs(self.model.coef_[0])
        else:
            raise ValueError("❌ Model nie obsługuje feature importance")
        
        if self.feature_columns:
            feature_importance = dict(zip(self.feature_columns, importance))
            # Sort by importance
            sorted_importance = dict(sorted(feature_importance.items(), 
                                         key=lambda x: x[1], reverse=True))
            return sorted_importance
        
        return importance.tolist()