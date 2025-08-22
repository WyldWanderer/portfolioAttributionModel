AI assisted project to create a portfolio attribution tool using a combination of Claude and OpenAI tools. 

You can see this live at https://huggingface.co/spaces/WyldWanderer/portfolioAttributionPrototype

WORK IN PROGRESS
1. Adding more models for different portfolio sizes
2. Cleanup of diagnostic logging

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
