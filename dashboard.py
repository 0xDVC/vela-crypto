import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from main import prepare_dashboard_data

def load_data():
    """Load or prepare data for the dashboard"""
    if 'symbols_data' not in st.session_state:
        with st.spinner('Preparing dashboard data... This may take a few minutes...'):
            symbols_data, predictions, backtest_results = prepare_dashboard_data()
            if symbols_data is not None:
                st.session_state['symbols_data'] = symbols_data
                st.session_state['predictions'] = predictions
                st.session_state['backtest_results'] = backtest_results
                st.success('Data loaded successfully!')
            else:
                st.error("Failed to load data. Please try again.")
                return None, None, None
    
    return (st.session_state['symbols_data'], 
            st.session_state['predictions'], 
            st.session_state['backtest_results'])

def create_dashboard():
    """Create the Streamlit dashboard"""
    st.title("Market Regime Analysis & Trading Results")
    
    # Load data
    symbols_data, predictions, backtest_results = load_data()
    if symbols_data is None:
        return
    
    # Regime Analysis Section
    st.header("Market Regime Analysis")
    
    # Create regime distribution chart
    regime_data = []
    for symbol, pred_df in predictions.items():
        if 'state' in pred_df.columns:  # Ensure state column exists
            regime_counts = pred_df['state'].value_counts(normalize=True) * 100
            regime_data.append({
                'Symbol': symbol,
                'Bull': regime_counts.get('bull', 0),
                'Bear': regime_counts.get('bear', 0),
                'Neutral': regime_counts.get('neutral', 0)
            })
    
    if regime_data:  # Only create chart if we have data
        regime_df = pd.DataFrame(regime_data)
        regime_df = regime_df.set_index('Symbol')
        st.bar_chart(regime_df)
    else:
        st.warning("No regime data available for visualization")
    
    # Symbol Selection
    if symbols_data:
        selected_symbol = st.selectbox("Select Symbol", list(symbols_data.keys()))
        
        # Create interactive chart
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.5, 0.3, 0.2])
        
        # Price and Portfolio Value
        if selected_symbol in symbols_data:
            fig.add_trace(
                go.Scatter(x=symbols_data[selected_symbol].index,
                          y=symbols_data[selected_symbol]['close'],
                          name="Price",
                          line=dict(color='gray', width=1)),
                row=1, col=1
            )
            
            if selected_symbol in backtest_results:
                fig.add_trace(
                    go.Scatter(x=backtest_results[selected_symbol]['portfolio'].index,
                              y=backtest_results[selected_symbol]['portfolio']['value'],
                              name="Portfolio Value",
                              line=dict(color='blue', width=2)),
                    row=1, col=1
                )
        
                # Market Regimes
                if selected_symbol in predictions:
                    regime_colors = {'bull': 'green', 'bear': 'red', 'neutral': 'gray'}
                    for state in ['bull', 'bear', 'neutral']:
                        mask = predictions[selected_symbol]['state'] == state
                        if mask.any():
                            fig.add_trace(
                                go.Scatter(x=predictions[selected_symbol][mask].index,
                                          y=[0]*mask.sum(),
                                          mode='markers',
                                          name=state.capitalize(),
                                          marker=dict(color=regime_colors[state], size=10)),
                                row=2, col=1
                            )
                
                # Drawdown
                fig.add_trace(
                    go.Scatter(x=backtest_results[selected_symbol]['portfolio'].index,
                              y=backtest_results[selected_symbol]['portfolio']['drawdown'] * 100,
                              name="Drawdown %",
                              line=dict(color='red', width=1)),
                    row=3, col=1
                )
        
                fig.update_layout(height=800, title=f"{selected_symbol} Analysis")
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance Metrics
                st.header("Performance Metrics")
                metrics = backtest_results[selected_symbol]['performance']
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Return", f"{metrics['total_return']:.2f}%")
                col2.metric("Win Rate", f"{metrics['win_rate']*100:.2f}%")
                col3.metric("Total Trades", metrics['total_trades'])
            else:
                st.warning(f"No backtest results available for {selected_symbol}")
    else:
        st.warning("No symbols data available")

if __name__ == "__main__":
    create_dashboard()