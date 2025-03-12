# models/optimization_visualization.py
# author: px
# date: 2021-11-09

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationVisualizer:
    """
    Visualization utilities for hyperparameter optimization results
    """
    def __init__(self, results_dir: str):
        """
        Args:
            results_dir: Directory containing optimization results
        """
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "visualization"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Load optimization results
        self.load_results()
        
    def load_results(self):
        """Load and process optimization results"""
        # Load main optimization results
        with open(self.results_dir / "optimization_results.json", "r") as f:
            self.optimization_results = json.load(f)
            
        # Load individual trial results
        self.trial_results = []
        for file in self.results_dir.glob("trial_*_results.json"):
            with open(file, "r") as f:
                self.trial_results.append(json.load(f))
                
        # Convert to DataFrame
        self.trials_df = pd.DataFrame([
            {
                "trial": tr["trial_number"],
                "score": tr["mean_score"],
                "std": tr["std_score"],
                **tr["parameters"]
            }
            for tr in self.trial_results
        ])
        
        logger.info(f"Loaded results for {len(self.trial_results)} trials")
    
    def plot_optimization_history(self, save: bool = True) -> go.Figure:
        """Plot optimization history with confidence intervals"""
        fig = go.Figure()
        
        # Sort trials by number
        df = self.trials_df.sort_values("trial")
        
        # Plot mean score
        fig.add_trace(go.Scatter(
            x=df.trial,
            y=df.score,
            mode='lines+markers',
            name='Mean CV Score',
            line=dict(color='blue'),
            marker=dict(size=8)
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=df.trial,
            y=df.score + df.std,
            mode='lines',
            name='Upper CI',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=df.trial,
            y=df.score - df.std,
            mode='lines',
            name='Lower CI',
            fill='tonexty',
            line=dict(width=0),
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title="Optimization History with Confidence Intervals",
            xaxis_title="Trial Number",
            yaxis_title="Cross-validation Score",
            template="plotly_white"
        )
        
        if save:
            fig.write_html(self.plots_dir / "optimization_history.html")
            
        return fig
    
    def plot_parameter_importance(self, save: bool = True) -> go.Figure:
        """Plot parameter importance based on correlation with score"""
        # Calculate correlations
        param_cols = [col for col in self.trials_df.columns 
                     if col not in ['trial', 'score', 'std']]
        correlations = []
        
        for param in param_cols:
            corr = np.corrcoef(self.trials_df[param], self.trials_df.score)[0, 1]
            correlations.append({
                'parameter': param,
                'correlation': abs(corr)  # Use absolute correlation
            })
            
        # Create bar plot
        df_corr = pd.DataFrame(correlations)
        df_corr = df_corr.sort_values('correlation', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=df_corr.correlation,
            y=df_corr.parameter,
            orientation='h'
        ))
        
        fig.update_layout(
            title="Parameter Importance (Absolute Correlation with Score)",
            xaxis_title="Absolute Correlation",
            yaxis_title="Parameter",
            template="plotly_white"
        )
        
        if save:
            fig.write_html(self.plots_dir / "parameter_importance.html")
            
        return fig
    
    def plot_parallel_coordinates(self, save: bool = True) -> go.Figure:
        """Plot parallel coordinates for parameter relationships"""
        # Normalize parameters for visualization
        df_norm = self.trials_df.copy()
        
        for col in df_norm.columns:
            if col not in ['trial', 'score', 'std']:
                df_norm[col] = (df_norm[col] - df_norm[col].min()) / \
                              (df_norm[col].max() - df_norm[col].min())
        
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=df_norm.score,
                colorscale='Viridis',
            ),
            dimensions=[
                dict(range=[0, 1],
                     label=col,
                     values=df_norm[col])
                for col in df_norm.columns
                if col not in ['trial', 'std']
            ]
        ))
        
        fig.update_layout(
            title="Parallel Coordinates Plot of Parameters",
            template="plotly_white"
        )
        
        if save:
            fig.write_html(self.plots_dir / "parallel_coordinates.html")
            
        return fig
    
    def plot_parameter_distributions(self, save: bool = True) -> go.Figure:
        """Plot distributions of parameters for best trials"""
        # Get top 10% of trials
        n_best = max(int(len(self.trials_df) * 0.1), 1)
        best_trials = self.trials_df.nsmallest(n_best, 'score')
        
        param_cols = [col for col in self.trials_df.columns 
                     if col not in ['trial', 'score', 'std']]
        n_params = len(param_cols)
        
        # Create subplots
        n_rows = (n_params + 1) // 2
        fig = make_subplots(rows=n_rows, cols=2,
                           subplot_titles=param_cols)
        
        for i, param in enumerate(param_cols):
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Add histogram for all trials
            fig.add_trace(
                go.Histogram(
                    x=self.trials_df[param],
                    name='All Trials',
                    opacity=0.7,
                    nbinsx=20
                ),
                row=row, col=col
            )
            
            # Add histogram for best trials
            fig.add_trace(
                go.Histogram(
                    x=best_trials[param],
                    name='Best Trials',
                    opacity=0.7,
                    nbinsx=20
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Parameter Distributions (All vs Best Trials)",
            showlegend=True,
            template="plotly_white",
            height=300 * n_rows
        )
        
        if save:
            fig.write_html(self.plots_dir / "parameter_distributions.html")
            
        return fig
    
    def plot_learning_curves(self, save: bool = True) -> go.Figure:
        """Plot learning curves for best trials"""
        # Get paths of best trial folders
        best_trial_num = min(self.trials_df.trial, key=lambda x: 
                           self.trials_df[self.trials_df.trial == x].score.iloc[0])
        best_trial_dir = self.results_dir / f"trial_{best_trial_num}"
        
        fig = go.Figure()
        
        # Load and plot learning curves for each fold
        for fold_dir in best_trial_dir.glob("fold_*"):
            with open(fold_dir / "training_results.json", "r") as f:
                fold_results = json.load(f)
                
            fold_num = int(fold_dir.name.split("_")[1])
            
            # Plot training loss
            fig.add_trace(go.Scatter(
                x=list(range(len(fold_results["history"]["train_loss"]))),
                y=fold_results["history"]["train_loss"],
                name=f"Fold {fold_num} Train",
                line=dict(dash='dash')
            ))
            
            # Plot validation loss
            fig.add_trace(go.Scatter(
                x=list(range(len(fold_results["history"]["val_loss"]))),
                y=fold_results["history"]["val_loss"],
                name=f"Fold {fold_num} Val"
            ))
        
        fig.update_layout(
            title="Learning Curves for Best Trial",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white"
        )
        
        if save:
            fig.write_html(self.plots_dir / "learning_curves.html")
            
        return fig
    
    def create_optimization_report(self):
        """Create comprehensive optimization report"""
        logger.info("Generating optimization report...")
        
        # Create all plots
        self.plot_optimization_history()
        self.plot_parameter_importance()
        self.plot_parallel_coordinates()
        self.plot_parameter_distributions()
        self.plot_learning_curves()
        
        # Create HTML report
        report_path = self.plots_dir / "optimization_report.html"
        
        with open(report_path, "w") as f:
            f.write(f"""
            <html>
            <head>
                <title>HAMF Optimization Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .plot {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>HAMF Optimization Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Best Parameters:</h2>
                <pre>{json.dumps(self.optimization_results["best_parameters"], indent=4)}</pre>
                
                <h2>Optimization History</h2>
                <div class="plot">
                    <iframe src="optimization_history.html" width="100%" height="600px" frameborder="0"></iframe>
                </div>
                
                <h2>Parameter Importance</h2>
                <div class="plot">
                    <iframe src="parameter_importance.html" width="100%" height="600px" frameborder="0"></iframe>
                </div>
                
                <h2>Parameter Relationships</h2>
                <div class="plot">
                    <iframe src="parallel_coordinates.html" width="100%" height="600px" frameborder="0"></iframe>
                </div>
                
                <h2>Parameter Distributions</h2>
                <div class="plot">
                    <iframe src="parameter_distributions.html" width="100%" height="600px" frameborder="0"></iframe>
                </div>
                
                <h2>Learning Curves</h2>
                <div class="plot">
                    <iframe src="learning_curves.html" width="100%" height="600px" frameborder="0"></iframe>
                </div>
            </body>
            </html>
            """)
        
        logger.info(f"Report generated at {report_path}")

# Example usage
def main():
    # Initialize visualizer
    visualizer = OptimizationVisualizer("path/to/optimization_results")
    
    # Generate comprehensive report
    visualizer.create_optimization_report()

if __name__ == "__main__":
    main()