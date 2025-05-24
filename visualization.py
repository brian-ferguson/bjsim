import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict

class Visualizer:
    """
    Handles all visualization for the blackjack simulation tool.
    Creates professional-grade charts using Plotly.
    """
    
    def __init__(self):
        # Default color scheme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def plot_expected_value_over_time(self, hours_played: int, hourly_ev: float, 
                                    hourly_std: float) -> go.Figure:
        """
        Plot expected value over time with confidence intervals.
        """
        hours = np.arange(0, hours_played + 1)
        cumulative_ev = hours * hourly_ev
        
        # Calculate confidence intervals (95%)
        cumulative_std = np.sqrt(hours) * hourly_std
        upper_ci = cumulative_ev + 1.96 * cumulative_std
        lower_ci = cumulative_ev - 1.96 * cumulative_std
        
        fig = go.Figure()
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=np.concatenate([hours, hours[::-1]]),
            y=np.concatenate([upper_ci, lower_ci[::-1]]),
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            showlegend=True
        ))
        
        # Add expected value line
        fig.add_trace(go.Scatter(
            x=hours,
            y=cumulative_ev,
            mode='lines',
            name='Expected Value',
            line=dict(color=self.colors['primary'], width=3)
        ))
        
        # Add break-even line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     annotation_text="Break Even")
        
        fig.update_layout(
            title='Expected Value Over Time',
            xaxis_title='Hours Played',
            yaxis_title='Cumulative Profit ($)',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def plot_bankroll_distribution(self, simulation_results: Dict) -> go.Figure:
        """
        Plot distribution of final bankrolls from Monte Carlo simulation.
        """
        final_bankrolls = simulation_results['final_bankrolls']
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=final_bankrolls,
            nbinsx=50,
            name='Final Bankroll Distribution',
            marker_color=self.colors['primary'],
            opacity=0.7
        ))
        
        # Add vertical lines for key statistics
        mean_bankroll = np.mean(final_bankrolls)
        median_bankroll = np.median(final_bankrolls)
        
        fig.add_vline(x=mean_bankroll, line_dash="dash", 
                     line_color=self.colors['success'],
                     annotation_text=f"Mean: ${mean_bankroll:.0f}")
        
        fig.add_vline(x=median_bankroll, line_dash="dot", 
                     line_color=self.colors['warning'],
                     annotation_text=f"Median: ${median_bankroll:.0f}")
        
        fig.update_layout(
            title='Distribution of Final Bankrolls',
            xaxis_title='Final Bankroll ($)',
            yaxis_title='Frequency',
            showlegend=False
        )
        
        return fig
    
    def plot_sample_trajectories(self, simulation_results: Dict, 
                               hours_played: int, num_samples: int = 10) -> go.Figure:
        """
        Plot sample bankroll trajectories over time.
        """
        trajectories = simulation_results['trajectories']
        
        # Randomly sample trajectories
        indices = np.random.choice(len(trajectories), 
                                 size=min(num_samples, len(trajectories)), 
                                 replace=False)
        
        fig = go.Figure()
        
        hours = np.arange(0, hours_played + 1)
        
        for i, idx in enumerate(indices):
            trajectory = trajectories[idx]
            
            # Determine color based on outcome
            final_bankroll = trajectory[-1]
            starting_bankroll = trajectory[0]
            
            if final_bankroll <= 0:
                color = self.colors['danger']
                opacity = 0.8
            elif final_bankroll > starting_bankroll:
                color = self.colors['success']
                opacity = 0.6
            else:
                color = self.colors['warning']
                opacity = 0.6
            
            fig.add_trace(go.Scatter(
                x=hours[:len(trajectory)],
                y=trajectory,
                mode='lines',
                name=f'Simulation {idx + 1}',
                line=dict(color=color, width=2),
                opacity=opacity,
                showlegend=False
            ))
        
        # Add starting bankroll line
        starting_bankroll = trajectories[0][0]
        fig.add_hline(y=starting_bankroll, line_dash="dash", 
                     line_color="gray",
                     annotation_text="Starting Bankroll")
        
        fig.update_layout(
            title=f'Sample Bankroll Trajectories ({num_samples} simulations)',
            xaxis_title='Hours Played',
            yaxis_title='Bankroll ($)',
            hovermode='x unified'
        )
        
        return fig
    
    def plot_risk_analysis(self, simulation_results: Dict) -> go.Figure:
        """
        Create a comprehensive risk analysis plot.
        """
        profits = simulation_results['profits']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Profit/Loss Distribution',
                'Probability Analysis',
                'Drawdown Analysis',
                'Risk Metrics'
            ),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "box"}, {"type": "indicator"}]]
        )
        
        # Profit/Loss Distribution
        fig.add_trace(
            go.Histogram(x=profits, nbinsx=30, name='Profit Distribution'),
            row=1, col=1
        )
        
        # Probability Analysis
        prob_profit = (profits > 0).mean() * 100
        prob_loss = (profits < 0).mean() * 100
        prob_breakeven = (np.abs(profits) < 100).mean() * 100
        
        fig.add_trace(
            go.Bar(
                x=['Profit', 'Loss', 'Break Even'],
                y=[prob_profit, prob_loss, prob_breakeven],
                name='Probabilities'
            ),
            row=1, col=2
        )
        
        # Drawdown Analysis (Box plot)
        drawdowns = []
        for trajectory in simulation_results['trajectories']:
            max_bankroll = np.maximum.accumulate(trajectory)
            drawdown = np.array(trajectory) - max_bankroll
            drawdowns.extend(drawdown[drawdown < 0])
        
        fig.add_trace(
            go.Box(y=drawdowns, name='Drawdowns'),
            row=2, col=1
        )
        
        # Risk Metrics (Gauge)
        risk_score = min(100, abs(np.min(profits)) / np.std(profits) * 10)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={'text': "Risk Score"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 30], 'color': "lightgray"},
                           {'range': [30, 70], 'color': "gray"},
                           {'range': [70, 100], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 80}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Comprehensive Risk Analysis",
            showlegend=False
        )
        
        return fig
    
    def plot_single_trajectory_vs_expected(self, actual_trajectory: List[float], 
                                         hours_played: int, hourly_ev: float, 
                                         starting_bankroll: float) -> go.Figure:
        """
        Plot a single actual trajectory against the expected value line.
        """
        hours = np.arange(0, len(actual_trajectory))
        expected_trajectory = starting_bankroll + (hours * hourly_ev)
        
        fig = go.Figure()
        
        # Add expected trajectory
        fig.add_trace(go.Scatter(
            x=hours,
            y=expected_trajectory,
            mode='lines',
            name='Expected Value',
            line=dict(color=self.colors['primary'], width=3, dash='dash')
        ))
        
        # Add actual trajectory
        final_profit = actual_trajectory[-1] - starting_bankroll
        expected_profit = expected_trajectory[-1] - starting_bankroll
        
        # Color based on performance vs expected
        if final_profit > expected_profit:
            color = self.colors['success']
            name = 'Actual (Above Expected)'
        elif final_profit < expected_profit:
            color = self.colors['danger'] 
            name = 'Actual (Below Expected)'
        else:
            color = self.colors['warning']
            name = 'Actual (At Expected)'
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=actual_trajectory,
            mode='lines',
            name=name,
            line=dict(color=color, width=3)
        ))
        
        # Add starting bankroll line
        fig.add_hline(y=starting_bankroll, line_dash="dot", 
                     line_color="gray",
                     annotation_text="Starting Bankroll")
        
        fig.update_layout(
            title='Actual vs Expected Performance',
            xaxis_title='Hours Played',
            yaxis_title='Bankroll ($)',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def plot_average_trajectory_vs_expected(self, avg_trajectory: List[float], 
                                          hours_played: int, hourly_ev: float, 
                                          starting_bankroll: float) -> go.Figure:
        """
        Plot the average trajectory of all simulations against the expected value line.
        """
        hours = np.arange(0, len(avg_trajectory))
        expected_trajectory = starting_bankroll + (hours * hourly_ev)
        
        fig = go.Figure()
        
        # Add expected trajectory
        fig.add_trace(go.Scatter(
            x=hours,
            y=expected_trajectory,
            mode='lines',
            name='Expected Value',
            line=dict(color=self.colors['primary'], width=3, dash='dash')
        ))
        
        # Add average actual trajectory
        final_avg_profit = avg_trajectory[-1] - starting_bankroll
        expected_profit = expected_trajectory[-1] - starting_bankroll
        
        # Color based on performance vs expected
        if final_avg_profit > expected_profit:
            color = self.colors['success']
            name = 'Average Actual (Above Expected)'
        elif final_avg_profit < expected_profit:
            color = self.colors['danger'] 
            name = 'Average Actual (Below Expected)'
        else:
            color = self.colors['warning']
            name = 'Average Actual (At Expected)'
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=avg_trajectory,
            mode='lines',
            name=name,
            line=dict(color=color, width=3)
        ))
        
        # Add starting bankroll line
        fig.add_hline(y=starting_bankroll, line_dash="dot", 
                     line_color="gray",
                     annotation_text="Starting Bankroll")
        
        fig.update_layout(
            title=f'Average Performance vs Expected ({len(avg_trajectory)} Hours)',
            xaxis_title='Hours Played',
            yaxis_title='Bankroll ($)',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def analyze_drawdowns(self, trajectories: List[List[float]]) -> Dict:
        """
        Analyze drawdowns across all simulation trajectories.
        """
        max_drawdowns = []
        
        for trajectory in trajectories:
            # Calculate running maximum
            running_max = np.maximum.accumulate(trajectory)
            
            # Calculate drawdowns
            drawdowns = np.array(trajectory) - running_max
            
            # Find maximum drawdown for this trajectory
            max_drawdown = np.min(drawdowns)
            max_drawdowns.append(abs(max_drawdown))
        
        return {
            'avg_max_drawdown': np.mean(max_drawdowns),
            'median_max_drawdown': np.median(max_drawdowns),
            'p95_drawdown': np.percentile(max_drawdowns, 95),
            'worst_drawdown': np.max(max_drawdowns),
            'std_drawdown': np.std(max_drawdowns)
        }
    
    def create_summary_dashboard(self, simulation_results: Dict, 
                               calculator_params: Dict) -> go.Figure:
        """
        Create a comprehensive dashboard with key metrics.
        """
        # Create subplots for dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Expected Value Progression',
                'Final Bankroll Distribution',
                'Risk Metrics',
                'Probability Analysis',
                'Parameter Summary',
                'Performance Statistics'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "table"}]
            ]
        )
        
        # This would be a comprehensive dashboard combining multiple visualizations
        # Implementation details would depend on specific requirements
        
        return fig
