import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from dataclasses import dataclass


@dataclass
class ReportConfig:
    figsize: tuple = (12, 8)
    style: str = "darkgrid"
    context: str = "talk"
    font_scale: float = 0.8


class ModelReport:
    def __init__(self, config: ReportConfig = None):
        self.config = config or ReportConfig()
        sns.set_style(self.config.style)
        sns.set_context(self.config.context, font_scale=self.config.font_scale)

    def plot_feature_distributions(self, features: pd.DataFrame, symbol: str):
        """Plot distributions of numerical features with statistics"""
        st.subheader(f"Feature Distributions - {symbol}")

        num_features = features.select_dtypes(include=np.number).columns
        features_subset = features[num_features]

        for feature in features_subset.columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            # plot distribution
            sns.histplot(
                features_subset[feature],
                bins=50,
                kde=True,
                ax=ax,
                color="blue",
                alpha=0.7,
            )

            # calculate statistics
            stats = {
                "Mean": features_subset[feature].mean(),
                "Median": features_subset[feature].median(),
                "Std": features_subset[feature].std(),
                "Skew": features_subset[feature].skew(),
            }

            # add stats box
            stats_text = "\n".join([f"{k}: {v:.2f}" for k, v in stats.items()])
            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                verticalalignment="top",
                horizontalalignment="right",
            )

            plt.title(f"{feature} distribution")
            st.pyplot(fig)
            plt.close()

    def plot_correlation_matrix(self, features: pd.DataFrame, symbol: str):
        """plot correlation heatmap of features"""
        st.subheader(f"Feature correlation matrix - {symbol}")

        fig, ax = plt.subplots(figsize=self.config.figsize)
        corr_matrix = features.corr()

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            cbar_kws={"label": "Correlation"},
        )

        plt.title(f"Feature correlation matrix - {symbol}")
        st.pyplot(fig)
        plt.close()

    def plot_trading_results(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        test_data: pd.DataFrame,
        predictions: pd.DataFrame,
        portfolio_values: pd.Series,
        buy_hold_values: pd.Series,
        metrics: Dict,
    ):
        """plot trading results with state predictions and performance metrics"""
        st.subheader(f"Trading results - {symbol}")

        # state distribution
        st.write("State distribution:")
        st.write(predictions["state"].value_counts())

        # trading visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # price and states
        ax1.plot(
            price_data.loc[test_data.index, "close"],
            label="Price",
            color="blue",
            alpha=0.6,
        )

        for state in ["bull", "bear", "neutral"]:
            mask = predictions["state"] == state
            color = {"bull": "green", "bear": "red", "neutral": "gray"}[state]
            ax1.fill_between(
                test_data.index,
                price_data.loc[test_data.index, "close"].min(),
                price_data.loc[test_data.index, "close"].max(),
                where=mask,
                color=color,
                alpha=0.2,
            )

        ax1.set_title(f"{symbol} Price and Market States")
        ax1.legend()

        # portfolio performance
        ax2.plot(
            portfolio_values.index, portfolio_values, label="Strategy", color="blue"
        )
        ax2.plot(
            buy_hold_values.index,
            buy_hold_values,
            label="Buy & Hold",
            color="gray",
            linestyle="--",
        )
        ax2.set_title("Portfolio Performance")
        ax2.legend()

        st.pyplot(fig)
        plt.close()

        # performance metrics
        st.subheader("Performance Metrics")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Strategy Metrics:")
            for metric, value in metrics["Portfolio"].items():
                st.write(f"{metric}: {value:.2f}")

        with col2:
            st.write("Buy & Hold Metrics:")
            for metric, value in metrics["Benchmark"].items():
                st.write(f"{metric}: {value:.2f}")


def create_streamlit_app(feature_dfs, price_data, test_data, predictions, results, metrics):
    """create Streamlit app for interactive visualization"""
    # get the single symbol we're analyzing
    symbol = list(feature_dfs.keys())[0]
    
    # feature analysis
    st.header(f"Analysis for {symbol}")
    
    report = ModelReport()
    features = feature_dfs[symbol]
    
    st.subheader("Feature Analysis")
    report.plot_feature_distributions(features, symbol)
    report.plot_correlation_matrix(features, symbol)
    
    # trading results
    st.subheader("Trading Results")
    report.plot_trading_results(
        symbol=symbol,
        price_data=price_data[symbol],
        test_data=test_data[symbol],
        predictions=predictions[symbol],
        portfolio_values=results[symbol]['portfolio_values'],
        buy_hold_values=results[symbol]['buy_hold_values'],
        metrics=metrics[symbol]
    )
