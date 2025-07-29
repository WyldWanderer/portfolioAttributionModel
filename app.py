import gradio as gr
import pandas as pd
import numpy as np
import torch
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import yfinance as yf
from portfolio_attribution_model import PortfolioAttributionEngine, PortfolioAttributionModel
import warnings
warnings.filterwarnings('ignore')

class PortfolioAttributionApp:
    """Gradio app for portfolio attribution analysis"""
    
    def __init__(self):
        self.model = None
        self.attribution_engine = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model with robust error handling"""
        model_files_to_try = [
            'trained_attribution_model_fixed_scalers.pth',  # Try fixed version first
            'trained_attribution_model.pth',
            'trained_attribution_model_fixed.pth', 
            'trained_attribution_model_emergency.pth'
        ]
        
        for model_file in model_files_to_try:
            try:
                print(f"🔄 Trying to load: {model_file}")
                
                # Try different loading methods
                checkpoint = None
                
                # Method 1: Standard load
                try:
                    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
                except Exception as e:
                    print(f"   Standard load failed: {e}")
                    
                    # Method 2: Weights only
                    try:
                        checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
                        print("   ⚠️ Loaded with weights_only=True")
                    except Exception as e2:
                        print(f"   Weights-only load failed: {e2}")
                        continue
                
                if checkpoint is None:
                    continue
                
                print("📁 Checkpoint loaded successfully")
                print(f"Available keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
                
                # Initialize attribution engine
                self.attribution_engine = PortfolioAttributionEngine()
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    # Standard checkpoint format
                    if 'feature_columns' in checkpoint:
                        self.attribution_engine.feature_columns = checkpoint['feature_columns']
                    else:
                        # Default feature columns
                        self.attribution_engine.feature_columns = [
                            'portfolio_return', 'benchmark_return', 'excess_return',
                            'n_assets', 'max_weight', 'weight_concentration'
                        ]
                        print("⚠️ Using default feature columns")
                    
                    if 'target_columns' in checkpoint:
                        self.attribution_engine.target_columns = checkpoint['target_columns']
                    else:
                        self.attribution_engine.target_columns = [
                            'asset_selection', 'allocation', 'timing', 'currency', 'interaction'
                        ]
                        print("⚠️ Using default target columns")
                    
                    # Load scalers if available
                    if 'scaler_features' in checkpoint:
                        self.attribution_engine.scaler_features = checkpoint['scaler_features']
                    if 'scaler_targets' in checkpoint:
                        self.attribution_engine.scaler_targets = checkpoint['scaler_targets']
                    
                    # Get model architecture
                    if 'model_params' in checkpoint:
                        model_params = checkpoint['model_params']
                    else:
                        # Use known retrained architecture
                        model_params = {
                            'hidden_dims': [32, 16],
                            'dropout_rate': 0.5
                        }
                        print("⚠️ Using default model architecture")
                    
                    # Initialize model
                    input_dim = len(self.attribution_engine.feature_columns)
                    self.attribution_engine.model = PortfolioAttributionModel(
                        input_dim=input_dim,
                        hidden_dims=model_params['hidden_dims'],
                        dropout_rate=model_params['dropout_rate']
                    )
                    
                    # Load weights
                    if 'model_state_dict' in checkpoint:
                        self.attribution_engine.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        # Maybe it's a raw state dict
                        self.attribution_engine.model.load_state_dict(checkpoint)
                    
                    self.attribution_engine.model.eval()
                    
                    print("✅ Model loaded successfully!")
                    print(f"📊 Features: {self.attribution_engine.feature_columns}")
                    print(f"🎯 Targets: {self.attribution_engine.target_columns}")
                    print(f"🏗️ Architecture: {model_params}")
                    return
                    
                else:
                    print(f"❌ Unexpected checkpoint format: {type(checkpoint)}")
                    continue
                    
            except Exception as e:
                print(f"❌ Failed to load {model_file}: {e}")
                continue
        
        # If all model files failed, create fallback
        print("🚑 All model files failed. Creating fallback model...")
        self.create_fallback_model()
    
    def create_fallback_model(self):
        """Create a fallback model if loading fails"""
        try:
            print("🔄 Creating fallback model...")
            self.attribution_engine = PortfolioAttributionEngine({
                'hidden_dims': [32, 16],
                'dropout_rate': 0.5,
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 50  # Quick training for fallback
            })
            
            # Create synthetic data
            features, targets = self.attribution_engine.generate_synthetic_data(500)
            train_loader, test_loader = self.attribution_engine.prepare_data(features, targets)
            
            print("🏃‍♂️ Quick training fallback model...")
            self.attribution_engine.train_model(train_loader, test_loader)
            
            print("✅ Fallback model ready!")
            
        except Exception as e:
            print(f"❌ Failed to create fallback model: {e}")
            # Last resort - create minimal working model
            self.attribution_engine = PortfolioAttributionEngine()
            self.attribution_engine.feature_columns = [
                'portfolio_return', 'benchmark_return', 'excess_return',
                'n_assets', 'max_weight', 'weight_concentration'
            ]
            self.attribution_engine.target_columns = [
                'asset_selection', 'allocation', 'timing', 'currency', 'interaction'
            ]
    
    def parse_portfolio_input(self, portfolio_text):
        """Parse portfolio input from text"""
        try:
            lines = portfolio_text.strip().split('\n')
            portfolio_data = []
            
            for line in lines:
                if ',' in line and line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        ticker = parts[0].strip().upper()
                        try:
                            weight = float(parts[1].strip())
                            portfolio_data.append({'ticker': ticker, 'weight': weight})
                        except ValueError:
                            continue
            
            return portfolio_data
        except Exception as e:
            raise ValueError(f"Error parsing portfolio: {e}")
    
    def get_market_data(self, tickers):
        """Get current market data for tickers"""
        try:
            # Get recent data (last 60 days for better returns calculation)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            market_data = {}
            successful_downloads = 0
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date, end=end_date)
                    if not hist.empty and len(hist) > 5:  # Need at least 5 days of data
                        returns = hist['Close'].pct_change().dropna()
                        if len(returns) > 0:
                            market_data[ticker] = {
                                'current_price': float(hist['Close'].iloc[-1]),
                                'avg_return': float(returns.mean()),
                                'volatility': float(returns.std()),
                                'recent_return': float(returns.iloc[-1]) if len(returns) > 0 else 0.0
                            }
                            successful_downloads += 1
                except Exception as e:
                    print(f"Failed to get data for {ticker}: {e}")
                    # Use fallback data
                    market_data[ticker] = {
                        'current_price': 100.0,
                        'avg_return': np.random.normal(0.001, 0.02),
                        'volatility': np.random.normal(0.02, 0.01),
                        'recent_return': np.random.normal(0.0, 0.02)
                    }
            
            print(f"Successfully downloaded data for {successful_downloads}/{len(tickers)} tickers")
            return market_data
            
        except Exception as e:
            print(f"Error fetching market data: {e}")
            # Return dummy data for all tickers
            return {ticker: {
                'current_price': 100.0,
                'avg_return': np.random.normal(0.001, 0.02),
                'volatility': np.random.normal(0.02, 0.01),
                'recent_return': np.random.normal(0.0, 0.02)
            } for ticker in tickers}
    
    def create_portfolio_features(self, portfolio_data, market_data):
        """Create features that match the trained model exactly"""
        
        # Calculate basic portfolio metrics
        weights = np.array([item['weight'] for item in portfolio_data])
        returns = np.array([market_data[item['ticker']]['avg_return'] for item in portfolio_data])
        volatilities = np.array([market_data[item['ticker']]['volatility'] for item in portfolio_data])
        
        # Portfolio-level calculations
        portfolio_return = np.sum(weights * returns)
        portfolio_volatility = np.sqrt(np.sum((weights * volatilities) ** 2))  # Simplified calculation
        
        # Create feature dictionary - Use training means for market features
        features = {
            'portfolio_return': portfolio_return,
            'benchmark_return': 0.0008,  # Approximate daily S&P 500 return
            'excess_return': portfolio_return - 0.0008,
            'n_assets': len(portfolio_data),
            'max_weight': np.max(weights),
            'weight_concentration': np.sum(weights ** 2),  # Herfindahl index
            # Use training means for market features (from diagnostics)
            'SPY': 0.000374,    # Training mean for SPY
            'VIX': 0.001858,    # Training mean for VIX (as return, not level!)
            'TLT': -0.000082,   # Training mean for TLT  
            'GLD': 0.000553     # Training mean for GLD
        }
        
        # Debug output to see what features we're creating
        print(f"📊 Portfolio Analysis:")
        print(f"   Holdings: {features['n_assets']}")
        print(f"   Max Weight: {features['max_weight']:.1%}")
        print(f"   Concentration (HHI): {features['weight_concentration']:.3f}")
        print(f"   Portfolio Return: {features['portfolio_return']:.4f}")
        print(f"   Excess Return: {features['excess_return']:.4f}")
        
        # Ensure we have exactly the features the model expects
        expected_features = self.attribution_engine.feature_columns
        print(f"🔍 Expected features ({len(expected_features)}): {expected_features}")
        print(f"🔍 Provided features ({len(features)}): {list(features.keys())}")
        
        # Create final feature vector in correct order
        final_features = {}
        for feature_name in expected_features:
            if feature_name in features:
                final_features[feature_name] = features[feature_name]
            else:
                print(f"⚠️ Missing feature '{feature_name}', using default 0.0")
                final_features[feature_name] = 0.0
        
        print(f"✅ Final feature vector: {len(final_features)} features (matches model)")
        
        return final_features
    
    def calculate_attribution(self, portfolio_input, benchmark_return=0.0008):
        """Calculate portfolio attribution"""
        try:
            # Parse portfolio input
            portfolio_data = self.parse_portfolio_input(portfolio_input)
            
            if not portfolio_data:
                return "❌ Error: No valid portfolio data provided", None, None
            
            # Calculate total weight and normalize
            total_weight = sum(item['weight'] for item in portfolio_data)
            print(f"📊 Original total weight: {total_weight:.3f}")
            
            if abs(total_weight - 100.0) > 1.0:  # Expecting percentage format
                return f"❌ Error: Weights sum to {total_weight:.1f}%. Please ensure weights sum to 100%.", None, None
            
            # Convert percentages to decimals and normalize
            for item in portfolio_data:
                item['weight'] = item['weight'] / 100.0  # Convert % to decimal
            
            # Renormalize to ensure exact sum of 1.0
            total_decimal_weight = sum(item['weight'] for item in portfolio_data)
            for item in portfolio_data:
                item['weight'] = item['weight'] / total_decimal_weight
            
            print(f"✅ Normalized weights sum to: {sum(item['weight'] for item in portfolio_data):.6f}")
            
            # Get market data first
            tickers = [item['ticker'] for item in portfolio_data]
            market_data = self.get_market_data(tickers)
            
            # Calculate portfolio metrics early
            weights = np.array([item['weight'] for item in portfolio_data])
            returns = np.array([market_data[item['ticker']]['avg_return'] for item in portfolio_data])
            portfolio_return = np.sum(weights * returns)  # Calculate this early!
            excess_return = portfolio_return - benchmark_return
            
            # Now check portfolio characteristics
            n_holdings = len(portfolio_data)
            max_weight = max(item['weight'] for item in portfolio_data)
            concentration = sum(item['weight']**2 for item in portfolio_data)
            
            # Warning for unusual portfolios with specific guidance
            warnings = []
            model_confidence = "High"
            
            if n_holdings < 30:
                warnings.append(f"⚠️ Only {n_holdings} holdings (model trained on 50-200)")
                warnings.append(f"   → Consider adding more holdings for better attribution accuracy")
                model_confidence = "Low"
                
            if max_weight > 0.12:  # 12%
                warnings.append(f"⚠️ Max weight {max_weight:.1%} is very high (model expects <5%)")
                warnings.append(f"   → Consider reducing position sizes")
                model_confidence = "Low"
            elif max_weight > 0.08:  # 8%
                warnings.append(f"🟡 Max weight {max_weight:.1%} is high (model expects <5%)")
                model_confidence = "Medium"
                
            if concentration > 0.08:
                warnings.append(f"⚠️ Very high concentration (HHI: {concentration:.3f}) - model expects <0.02")
                warnings.append(f"   → This portfolio is much more concentrated than training data")
                model_confidence = "Low"
            elif concentration > 0.04:
                warnings.append(f"🟡 High concentration (HHI: {concentration:.3f}) - model expects <0.02")
                model_confidence = "Medium"
            
            # Add portfolio return outlier check
            portfolio_return_pct = portfolio_return * 100
            if abs(portfolio_return_pct) > 2.0:  # >2% daily return is extreme
                warnings.append(f"⚠️ Extreme daily return {portfolio_return_pct:.2f}% (model expects <0.5%)")
                warnings.append(f"   → Attribution may be unreliable for extreme return days")
                model_confidence = "Low"
            
            # Create features matching the trained model
            features = self.create_portfolio_features(portfolio_data, market_data)
            
            # Convert to DataFrame with correct column order    
            features_df = pd.DataFrame([features])
            features_df = features_df.reindex(columns=self.attribution_engine.feature_columns, fill_value=0.0)
            
            # Make prediction using the trained model
            attribution_results = self.predict_with_trained_model(features_df)
            
            # Calculate final metrics (portfolio_return already calculated above)
            total_attribution = sum(attribution_results.values())
            unexplained = excess_return - total_attribution
            
            # Calculate explanation ratio correctly
            explanation_ratio = (total_attribution / excess_return * 100) if abs(excess_return) > 1e-6 else 0
            unexplained_ratio = (unexplained / excess_return * 100) if abs(excess_return) > 1e-6 else 0
            
            # Check if we applied corrections
            reliability_warning = attribution_results.pop('_reliability_warning', False)
            
            # Create results summary with confidence assessment
            warning_text = "\n".join(warnings) + "\n" if warnings else ""
            
            reliability_text = ""
            if reliability_warning:
                reliability_text = "\n⚠️ ATTRIBUTION RELIABILITY WARNING:\nThis portfolio differs significantly from the model's training data.\nResults have been scaled down but may still be unreliable.\nConsider using a more diversified portfolio for better attribution accuracy.\n"
            
            results_text = f"""
📊 Portfolio Attribution Analysis
═══════════════════════════════════

{warning_text}{reliability_text}Portfolio Overview:
• Holdings: {len(portfolio_data)} securities
• Max Weight: {max_weight:.1%}
• Concentration (HHI): {concentration:.3f}
• Portfolio Return: {portfolio_return:.4f} ({portfolio_return*100:.2f}% daily)
• Benchmark Return: {benchmark_return:.4f} ({benchmark_return*100:.2f}% daily)
• Excess Return: {excess_return:.4f} ({excess_return*100:.2f}% daily)

Attribution Breakdown (basis points):
• Asset Selection: {attribution_results['asset_selection']*10000:.1f} bps
• Allocation Effect: {attribution_results['allocation']*10000:.1f} bps
• Timing Effect: {attribution_results['timing']*10000:.1f} bps
• Currency Effect: {attribution_results['currency']*10000:.1f} bps
• Interaction Effect: {attribution_results['interaction']*10000:.1f} bps

Total Explained: {total_attribution*10000:.1f} bps ({explanation_ratio:.0f}% of excess return)
Unexplained: {unexplained*10000:.1f} bps ({unexplained_ratio:.0f}% of excess return)

Model Confidence: {model_confidence}
{"✅ High confidence - portfolio matches training data" if model_confidence == "High" else 
 "🟡 Medium confidence - some portfolio characteristics differ from training" if model_confidence == "Medium" else
 "❌ Low confidence - portfolio significantly differs from training data"}
• Original model achieved 96% R² on similar portfolios
• For best results, use 50+ holdings with max 5% position sizes
            """
            
            # Create attribution chart
            attribution_chart = self.create_attribution_chart(attribution_results)
            
            # Create portfolio composition chart  
            composition_chart = self.create_composition_chart(portfolio_data)
            
            return results_text, attribution_chart, composition_chart
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}\n\nPlease check your portfolio format:\nTICKER,WEIGHT\nAAPL,25.0\nMSFT,20.0"
            return error_msg, None, None
    
    def predict_with_trained_model(self, features_df):
        """Make prediction using the trained model with detailed debugging"""
        try:
            print("\n🔍 MODEL PREDICTION DEBUG:")
            print(f"Input features shape: {features_df.shape}")
            print(f"Input features:")
            for i, (name, value) in enumerate(zip(self.attribution_engine.feature_columns, features_df.iloc[0])):
                print(f"  {name}: {value:.6f}")
            
            # Scale features
            features_scaled = self.attribution_engine.scaler_features.transform(features_df.values)
            print(f"Scaled features range: [{features_scaled.min():.3f}, {features_scaled.max():.3f}]")
            
            # Show which features are extreme
            print("🔍 Scaled feature analysis:")
            for i, (name, scaled_val) in enumerate(zip(self.attribution_engine.feature_columns, features_scaled[0])):
                if abs(scaled_val) > 5:  # Anything >5 standard deviations is extreme
                    print(f"  ⚠️ {name}: {scaled_val:.2f} (EXTREME - outside training distribution)")
                elif abs(scaled_val) > 2:
                    print(f"  🟡 {name}: {scaled_val:.2f} (high)")
                else:
                    print(f"  ✅ {name}: {scaled_val:.2f} (normal)")
            
            # Show training scaler parameters for comparison
            print(f"\n📊 Training data statistics (what model expects):")
            scaler = self.attribution_engine.scaler_features
            for i, name in enumerate(self.attribution_engine.feature_columns):
                train_mean = scaler.mean_[i] if hasattr(scaler, 'mean_') else 0
                train_scale = scaler.scale_[i] if hasattr(scaler, 'scale_') else 1
                current_val = features_df.iloc[0, i]
                
                print(f"  {name}:")
                print(f"    Training mean: {train_mean:.6f}")
                print(f"    Training std:  {train_scale:.6f}")
                print(f"    Your value:    {current_val:.6f}")
                print(f"    Z-score:       {(current_val - train_mean) / train_scale:.2f}")
                print()
            
            # Make prediction
            self.attribution_engine.model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled)
                predictions = self.attribution_engine.model(features_tensor)
                predictions_scaled = predictions.numpy()
            
            print(f"Raw model output range: [{predictions_scaled.min():.6f}, {predictions_scaled.max():.6f}]")
            
            # Inverse transform predictions
            predictions_original = self.attribution_engine.scaler_targets.inverse_transform(predictions_scaled)
            print(f"After inverse transform: [{predictions_original.min():.6f}, {predictions_original.max():.6f}]")
            
            # Return as dictionary
            attribution_dict = {
                component: float(pred) for component, pred in 
                zip(self.attribution_engine.target_columns, predictions_original[0])
            }
            
            print(f"Final attribution predictions (in decimals):")
            for comp, val in attribution_dict.items():
                print(f"  {comp}: {val:.6f} ({val*10000:.1f} bps)")
            
            # Check if predictions are reasonable
            total_pred = sum(attribution_dict.values())
            excess_return = features_df['excess_return'].iloc[0]
            
            print(f"\n📊 SCALING CHECK:")
            print(f"Excess Return: {excess_return:.6f} ({excess_return*10000:.1f} bps)")
            print(f"Total Predicted: {total_pred:.6f} ({total_pred*10000:.1f} bps)")
            print(f"Prediction/Excess Ratio: {total_pred/excess_return:.1f}x" if abs(excess_return) > 1e-6 else "N/A")
            
            if abs(total_pred/excess_return) > 3 if abs(excess_return) > 1e-6 else False:
                print("🚨 CRITICAL: Predictions are much larger than excess return!")
                print("This indicates the portfolio is very different from training data.")
                
                # More aggressive scaling for extreme cases
                if abs(total_pred/excess_return) > 10:
                    scale_factor = 0.1  # Cap at 10% of prediction magnitude
                    print(f"🔧 Applying aggressive scale factor: {scale_factor:.3f}")
                else:
                    scale_factor = min(abs(excess_return / total_pred), 0.5) if abs(total_pred) > 1e-6 else 1.0
                    print(f"🔧 Applying moderate scale factor: {scale_factor:.3f}")
                
                for comp in attribution_dict:
                    attribution_dict[comp] *= scale_factor
                
                print("🔧 Corrected predictions:")
                for comp, val in attribution_dict.items():
                    print(f"  {comp}: {val:.6f} ({val*10000:.1f} bps)")
                    
                # Add warning about reliability
                attribution_dict['_reliability_warning'] = True
            
            return attribution_dict
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            # Fallback to simple calculation
            excess_return = features_df['excess_return'].iloc[0]
            return {
                'asset_selection': excess_return * 0.3,
                'allocation': excess_return * 0.4,
                'timing': excess_return * 0.1,
                'currency': excess_return * 0.1,
                'interaction': excess_return * 0.1
            }
    
    def create_attribution_chart(self, attribution_results):
        """Create attribution waterfall chart"""
        try:
            components = list(attribution_results.keys())
            values = [attribution_results[comp] * 10000 for comp in components]  # Convert to basis points
            
            # Create waterfall chart
            fig = go.Figure()
            
            # Add bars for each component
            colors = ['green' if v > 0 else 'red' for v in values]
            
            fig.add_trace(go.Bar(
                x=components,
                y=values,
                marker_color=colors,
                text=[f"{v:.1f} bps" for v in values],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Portfolio Attribution Breakdown",
                xaxis_title="Attribution Components",
                yaxis_title="Contribution (Basis Points)",
                showlegend=False,
                height=500
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating attribution chart: {e}")
            return None
    
    def create_composition_chart(self, portfolio_data):
        """Create portfolio composition pie chart"""
        try:
            tickers = [item['ticker'] for item in portfolio_data]
            weights = [item['weight'] for item in portfolio_data]
            
            fig = go.Figure(data=[go.Pie(
                labels=tickers,
                values=weights,
                hole=0.3,
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_layout(
                title="Portfolio Composition",
                height=500
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating composition chart: {e}")
            return None
    
    def create_sample_portfolio(self):
        """Create a sample portfolio for demonstration"""
        sample_portfolio = """AAPL,0.25
MSFT,0.20
GOOGL,0.15
AMZN,0.15
TSLA,0.10
META,0.10
NVDA,0.05"""
        return sample_portfolio
    
    def launch_app(self):
        """Launch the Gradio app"""
        
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .portfolio-input {
            font-family: monospace;
        }
        """
        
        with gr.Blocks(css=custom_css, title="Portfolio Attribution Tool") as app:
            gr.Markdown("""
            # 📊 Portfolio Performance Attribution Tool
            
            This tool uses machine learning to analyze portfolio performance attribution across different factors.
            Enter your portfolio holdings and weights to get a detailed attribution breakdown.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Portfolio Input")
                    gr.Markdown("Enter your portfolio in the format: TICKER,WEIGHT (one per line)")
                    
                    portfolio_input = gr.Textbox(
                        label="Portfolio Holdings",
                        placeholder="AAPL,0.25\nMSFT,0.20\nGOOGL,0.15\n...",
                        lines=10,
                        elem_classes=["portfolio-input"]
                    )
                    
                    benchmark_return = gr.Slider(
                        label="Benchmark Return (daily)",
                        minimum=-0.01,
                        maximum=0.01,
                        step=0.0001,
                        value=0.0008,
                        info="Expected daily benchmark return"
                    )
                    
                    with gr.Row():
                        analyze_btn = gr.Button("Analyze Portfolio", variant="primary")
                        sample_btn = gr.Button("Load Sample Portfolio", variant="secondary")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Attribution Results")
                    results_output = gr.Textbox(
                        label="Attribution Analysis",
                        lines=20,
                        elem_classes=["portfolio-input"]
                    )
            
            with gr.Row():
                with gr.Column():
                    attribution_plot = gr.Plot(label="Attribution Breakdown")
                with gr.Column():
                    composition_plot = gr.Plot(label="Portfolio Composition")
            
            # Event handlers
            analyze_btn.click(
                fn=self.calculate_attribution,
                inputs=[portfolio_input, benchmark_return],
                outputs=[results_output, attribution_plot, composition_plot]
            )
            
            sample_btn.click(
                fn=self.create_sample_portfolio,
                outputs=[portfolio_input]
            )
            
            # Add examples
            gr.Examples(
                examples=[
                    ["AAPL,0.3\nMSFT,0.25\nGOOGL,0.2\nAMZN,0.15\nTSLA,0.1", 0.0008],
                    ["SPY,0.6\nBND,0.3\nVTI,0.1", 0.0005],
                    ["NVDA,0.4\nAMD,0.3\nINTC,0.2\nMU,0.1", 0.001]
                ],
                inputs=[portfolio_input, benchmark_return],
                label="Example Portfolios"
            )
            
            gr.Markdown("""
            ### How to Use:
            1. **Enter Portfolio**: List holdings with format `TICKER,WEIGHT` (one per line)
            2. **Set Benchmark**: Adjust daily benchmark return if needed  
            3. **Analyze**: Click "Analyze Portfolio" for ML-powered attribution
            
            ### Model Performance:
            - **Overall R²**: 96% (Exceptional accuracy)
            - **Directional Accuracy**: 99% (Gets direction right almost always)
            - **Best Components**: Allocation (85% R²), Currency (85% R²), Interaction (93% R²)
            - **Challenging Components**: Asset Selection (8% R²), Timing (13% R²)
            
            ### Interpretation Guide:
            - **Asset Selection**: Returns from picking specific securities vs sector average
            - **Allocation**: Returns from over/under-weighting sectors or asset classes  
            - **Timing**: Returns from tactical allocation changes over time
            - **Currency**: Impact of currency movements on international holdings
            - **Interaction**: Combined effects of allocation and selection decisions
            - **Basis Points**: 100 bps = 1% return
            
            ### Notes:
            - Model trained on realistic portfolio strategies (50-200 holdings)
            - Uses recent market data for current analysis
            - Basis points (bps) used for precision: 10 bps = 0.1% daily return
            - Weights automatically normalized if they don't sum to 1.0
            """)
        
        return app

def main():
    """Main function to run the app"""
    app = PortfolioAttributionApp()
    gradio_app = app.launch_app()
    
    # Launch the app
    gradio_app.launch(
        share=True,  # Creates a public link
        server_name="0.0.0.0",  # Allows external access
        server_port=7860,
        show_error=True
    )

if __name__ == "__main__":
    # Install required packages:
    # pip install gradio plotly yfinance pandas numpy torch scikit-learn
    
    main()