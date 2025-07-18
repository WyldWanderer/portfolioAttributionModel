import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class PortfolioAttributionDataset(Dataset):
    """Dataset class for portfolio attribution data"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class PortfolioAttributionModel(nn.Module):
    """Multi-task neural network for portfolio attribution"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], 
                 output_dim: int = 5, dropout_rate: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output heads for different attribution components
        self.asset_selection_head = nn.Linear(prev_dim, 1)
        self.allocation_head = nn.Linear(prev_dim, 1) 
        self.timing_head = nn.Linear(prev_dim, 1)
        self.currency_head = nn.Linear(prev_dim, 1)
        self.interaction_head = nn.Linear(prev_dim, 1)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        asset_selection = self.asset_selection_head(features)
        allocation = self.allocation_head(features)
        timing = self.timing_head(features)
        currency = self.currency_head(features)
        interaction = self.interaction_head(features)
        
        return torch.cat([asset_selection, allocation, timing, currency, interaction], dim=1)

class PortfolioAttributionEngine:
    """Main engine for portfolio attribution analysis"""
    
    def __init__(self, model_params: Dict = None):
        self.model_params = model_params or {
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100
        }
        self.model = None
        self.scaler_features = StandardScaler()
        self.scaler_targets = StandardScaler()
        self.feature_columns = None
        self.target_columns = ['asset_selection', 'allocation', 'timing', 'currency', 'interaction']
        
    def create_features(self, portfolio_data: pd.DataFrame) -> pd.DataFrame:
        """Create features from portfolio data"""
        features = pd.DataFrame()
        
        # Portfolio composition features
        features['portfolio_concentration'] = (portfolio_data['weights'] ** 2).sum()
        features['num_holdings'] = len(portfolio_data)
        features['max_weight'] = portfolio_data['weights'].max()
        features['min_weight'] = portfolio_data['weights'].min()
        features['weight_std'] = portfolio_data['weights'].std()
        
        # Return-based features
        features['portfolio_return'] = (portfolio_data['weights'] * portfolio_data['returns']).sum()
        features['return_volatility'] = portfolio_data['returns'].std()
        features['max_return'] = portfolio_data['returns'].max()
        features['min_return'] = portfolio_data['returns'].min()
        
        # Sector/region features (assuming these columns exist)
        if 'sector' in portfolio_data.columns:
            sector_weights = portfolio_data.groupby('sector')['weights'].sum()
            for i, sector in enumerate(sector_weights.index[:10]):  # Top 10 sectors
                features[f'sector_{sector}_weight'] = sector_weights.get(sector, 0)
        
        # Market factor features (assuming these columns exist)
        market_factors = ['market_return', 'risk_free_rate', 'vix', 'term_spread']
        for factor in market_factors:
            if factor in portfolio_data.columns:
                features[factor] = portfolio_data[factor].iloc[0]
        
        # Time-based features
        if 'date' in portfolio_data.columns:
            features['month'] = pd.to_datetime(portfolio_data['date']).dt.month.iloc[0]
            features['quarter'] = pd.to_datetime(portfolio_data['date']).dt.quarter.iloc[0]
        
        return features.fillna(0)
    
    def generate_synthetic_data(self, n_samples: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic portfolio data for demonstration"""
        np.random.seed(42)
        
        # Generate synthetic portfolio features
        data = []
        for _ in range(n_samples):
            n_assets = np.random.randint(10, 100)
            weights = np.random.dirichlet(np.ones(n_assets))
            returns = np.random.normal(0.08/252, 0.2/np.sqrt(252), n_assets)  # Daily returns
            
            # Market factors
            market_return = np.random.normal(0.1/252, 0.15/np.sqrt(252))
            risk_free_rate = np.random.normal(0.02/252, 0.005/np.sqrt(252))
            vix = np.random.normal(20, 5)
            term_spread = np.random.normal(1.5, 0.5)
            
            # Create portfolio data
            portfolio_df = pd.DataFrame({
                'weights': weights,
                'returns': returns,
                'market_return': market_return,
                'risk_free_rate': risk_free_rate,
                'vix': vix,
                'term_spread': term_spread,
                'sector': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer'], n_assets),
                'date': pd.Timestamp.now()
            })
            
            features = self.create_features(portfolio_df)
            
            # Generate synthetic attribution targets
            portfolio_return = (weights * returns).sum()
            benchmark_return = market_return
            excess_return = portfolio_return - benchmark_return
            
            # Synthetic attribution decomposition
            asset_selection = np.random.normal(0, 0.001) * excess_return
            allocation = np.random.normal(0, 0.001) * excess_return  
            timing = np.random.normal(0, 0.0005) * excess_return
            currency = np.random.normal(0, 0.0003) * excess_return
            interaction = excess_return - asset_selection - allocation - timing - currency
            
            targets = pd.DataFrame({
                'asset_selection': [asset_selection],
                'allocation': [allocation],
                'timing': [timing],
                'currency': [currency],
                'interaction': [interaction]
            })
            
            data.append((features, targets))
        
        # Combine all data
        all_features = pd.concat([item[0] for item in data], ignore_index=True)
        all_targets = pd.concat([item[1] for item in data], ignore_index=True)
        
        return all_features, all_targets
    
    def prepare_data(self, features: pd.DataFrame, targets: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Prepare data for training with better validation"""
        # Store feature columns for later use
        self.feature_columns = features.columns.tolist()
        
        # Check for invalid values
        print(f"Features shape: {features.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Features NaN count: {features.isnull().sum().sum()}")
        print(f"Targets NaN count: {targets.isnull().sum().sum()}")
        print(f"Features infinite count: {np.isinf(features.values).sum()}")
        print(f"Targets infinite count: {np.isinf(targets.values).sum()}")
        
        # Clean data more aggressively
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        targets = targets.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Remove extreme outliers (beyond 3 standard deviations)
        for col in targets.columns:
            target_std = targets[col].std()
            target_mean = targets[col].mean()
            outlier_mask = np.abs(targets[col] - target_mean) > 3 * target_std
            targets.loc[outlier_mask, col] = target_mean
            
        print(f"Target ranges after cleaning:")
        for col in targets.columns:
            print(f"  {col}: [{targets[col].min():.6f}, {targets[col].max():.6f}]")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        # Scale features and targets
        X_train_scaled = self.scaler_features.fit_transform(X_train)
        X_test_scaled = self.scaler_features.transform(X_test)
        
        y_train_scaled = self.scaler_targets.fit_transform(y_train)
        y_test_scaled = self.scaler_targets.transform(y_test)
        
        # Additional check after scaling
        print(f"Scaled targets range: [{y_train_scaled.min():.3f}, {y_train_scaled.max():.3f}]")
        
        # Create datasets
        train_dataset = PortfolioAttributionDataset(X_train_scaled, y_train_scaled)
        test_dataset = PortfolioAttributionDataset(X_test_scaled, y_test_scaled)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.model_params['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.model_params['batch_size'], shuffle=False)
        
        return train_loader, test_loader
    
    def train_model(self, train_loader: DataLoader, test_loader: DataLoader):
        """Train the attribution model"""
        input_dim = len(self.feature_columns)
        self.model = PortfolioAttributionModel(
            input_dim=input_dim,
            hidden_dims=self.model_params['hidden_dims'],
            dropout_rate=self.model_params['dropout_rate']
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.model_params['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        
        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        patience_counter = 0
        patience = 25  # Early stopping patience
        current_lr = self.model_params['learning_rate']
        
        for epoch in range(self.model_params['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch_features, batch_targets in test_loader:
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    test_loss += loss.item()
            
            train_loss /= len(train_loader)
            test_loss /= len(test_loader)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            # Check if learning rate was reduced
            old_lr = current_lr
            scheduler.step(test_loss)
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr != old_lr:
                print(f'Learning rate reduced to {current_lr:.6f} at epoch {epoch}')
            
            # Early stopping logic with safer saving
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
                
                # Save best model more safely
                try:
                    import os  # Import here since it's inside the class
                    temp_path = 'best_model_temp.pth'
                    final_path = 'best_model.pth'
                    
                    # Save to temporary file first
                    torch.save(self.model.state_dict(), temp_path)
                    
                    # Only move to final location if save succeeded
                    import os
                    if os.path.exists(temp_path):
                        if os.path.exists(final_path):
                            os.remove(final_path)  # Remove old version
                        os.rename(temp_path, final_path)  # Atomic rename
                        
                    print(f"💾 Best model saved at epoch {epoch} (loss: {best_test_loss:.6f})")
                    
                except Exception as save_error:
                    print(f"⚠️ Warning: Could not save best model: {save_error}")
                    
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, LR: {current_lr:.6f}')
                print(f'Best Test Loss: {best_test_loss:.6f}, Patience: {patience_counter}/{patience}')
            
            # Early stopping
            if patience_counter >= patience:
                print(f'🛑 Early stopping at epoch {epoch}')
                print(f'📊 Best performance was at epoch {epoch - patience} with loss {best_test_loss:.6f}')
                
                # Load best model safely
                try:
                    import os  # Import here too
                    if os.path.exists('best_model.pth'):
                        self.model.load_state_dict(torch.load('best_model.pth', weights_only=False))
                        print("✅ Loaded best model from early stopping")
                    else:
                        print("⚠️ Best model file not found, keeping current weights")
                except Exception as load_error:
                    print(f"⚠️ Could not load best model: {load_error}")
                break
        
        return train_losses, test_losses
    
    def predict_attribution(self, portfolio_data: pd.DataFrame) -> Dict[str, float]:
        """Predict attribution for a given portfolio"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Create features
        features = self.create_features(portfolio_data)
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0
        
        # Reorder columns to match training data
        features = features[self.feature_columns]
        
        # Scale features
        features_scaled = self.scaler_features.transform(features.values.reshape(1, -1))
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled)
            predictions = self.model(features_tensor)
            predictions_scaled = predictions.numpy()
        
        # Inverse transform predictions
        predictions_original = self.scaler_targets.inverse_transform(predictions_scaled)
        
        # Return as dictionary
        attribution_dict = {
            component: float(pred) for component, pred in 
            zip(self.target_columns, predictions_original[0])
        }
        
        return attribution_dict
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance with comprehensive metrics"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                outputs = self.model(batch_features)
                all_predictions.append(outputs.numpy())
                all_targets.append(batch_targets.numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # Inverse transform for evaluation
        predictions_original = self.scaler_targets.inverse_transform(predictions)
        targets_original = self.scaler_targets.inverse_transform(targets)
        
        # Calculate comprehensive metrics for each attribution component
        metrics = {}
        for i, component in enumerate(self.target_columns):
            # Standard regression metrics
            mse = mean_squared_error(targets_original[:, i], predictions_original[:, i])
            r2 = r2_score(targets_original[:, i], predictions_original[:, i])
            mae = np.mean(np.abs(targets_original[:, i] - predictions_original[:, i]))
            rmse = np.sqrt(mse)
            
            # Financial metrics
            actual = targets_original[:, i]
            predicted = predictions_original[:, i]
            
            # Mean Absolute Percentage Error (MAPE)
            mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
            
            # Directional Accuracy (what % of time we get the sign right)
            directional_accuracy = np.mean(np.sign(actual) == np.sign(predicted)) * 100
            
            # Correlation coefficient
            correlation = np.corrcoef(actual, predicted)[0, 1] if len(actual) > 1 else 0
            
            # Store all metrics
            metrics[f'{component}_mse'] = mse
            metrics[f'{component}_r2'] = r2
            metrics[f'{component}_mae'] = mae
            metrics[f'{component}_rmse'] = rmse
            metrics[f'{component}_mape'] = mape
            metrics[f'{component}_directional_accuracy'] = directional_accuracy
            metrics[f'{component}_correlation'] = correlation
        
        # Overall portfolio metrics
        total_actual = np.sum(targets_original, axis=1)
        total_predicted = np.sum(predictions_original, axis=1)
        
        metrics['portfolio_total_r2'] = r2_score(total_actual, total_predicted)
        metrics['portfolio_total_correlation'] = np.corrcoef(total_actual, total_predicted)[0, 1]
        metrics['portfolio_directional_accuracy'] = np.mean(np.sign(total_actual) == np.sign(total_predicted)) * 100
        
        return metrics

# Example usage
if __name__ == "__main__":
    # Initialize the attribution engine
    attribution_engine = PortfolioAttributionEngine()
    
    # Generate synthetic data
    print("Generating synthetic data...")
    features, targets = attribution_engine.generate_synthetic_data(n_samples=5000)
    
    # Prepare data
    print("Preparing data...")
    train_loader, test_loader = attribution_engine.prepare_data(features, targets)
    
    # Train model
    print("Training model...")
    train_losses, test_losses = attribution_engine.train_model(train_loader, test_loader)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = attribution_engine.evaluate_model(test_loader)
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # Example prediction
    print("\nExample Attribution Prediction:")
    sample_portfolio = pd.DataFrame({
        'weights': [0.3, 0.2, 0.15, 0.35],
        'returns': [0.001, -0.002, 0.003, 0.0015],
        'market_return': 0.0008,
        'risk_free_rate': 0.0001,
        'vix': 18.5,
        'term_spread': 1.2,
        'sector': ['Tech', 'Finance', 'Healthcare', 'Energy'],
        'date': pd.Timestamp.now()
    })
    
    attribution = attribution_engine.predict_attribution(sample_portfolio)
    print("Attribution Results:")
    for component, value in attribution.items():
        print(f"{component}: {value:.6f}")
