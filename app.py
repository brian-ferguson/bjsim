import streamlit as st
import numpy as np
import pandas as pd
from blackjack_calculator import BlackjackCalculator
from monte_carlo_simulation import MonteCarloSimulation
from visualization import Visualizer

def main():
    st.set_page_config(
        page_title="Blackjack Card Counting Simulator",
        page_icon="♠️",
        layout="wide"
    )
    
    st.title("♠️ Blackjack Card Counting Simulation Tool")
    st.markdown("**Analyze expected value, variance, and risk of ruin with Monte Carlo analysis**")
    
    # Sidebar for inputs
    st.sidebar.header("Simulation Parameters")
    
    # Input validation ranges
    deck_options = [1, 2, 6]
    
    # User inputs with validation
    num_decks = st.sidebar.selectbox(
        "Number of Decks in Shoe",
        options=deck_options,
        index=2,
        help="Standard casinos use 6-deck shoes"
    )
    
    bet_spread_col1, bet_spread_col2 = st.sidebar.columns(2)
    with bet_spread_col1:
        min_bet = st.number_input(
            "Min Bet (units)",
            min_value=1,
            max_value=10,
            value=1,
            step=1
        )
    
    with bet_spread_col2:
        max_bet = st.number_input(
            "Max Bet (units)",
            min_value=min_bet,
            max_value=50,
            value=12,
            step=1
        )
    
    starting_bankroll = st.sidebar.number_input(
        "Starting Bankroll ($)",
        min_value=100,
        max_value=100000,
        value=2000,
        step=100,
        help="Your initial bankroll in dollars"
    )
    
    hours_played = st.sidebar.number_input(
        "Hours Played",
        min_value=1,
        max_value=1000,
        value=100,
        step=10,
        help="Total hours of play to simulate"
    )
    
    win_rate = st.sidebar.slider(
        "Win Rate per 100 hands (%)",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Expected advantage based on counting system and conditions"
    )
    
    hands_per_hour = st.sidebar.number_input(
        "Hands per Hour",
        min_value=50,
        max_value=500,
        value=100,
        step=10,
        help="100 for solo play, 250+ for team play"
    )
    
    monte_carlo_runs = st.sidebar.number_input(
        "Monte Carlo Runs",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Number of simulations to run for variance analysis"
    )
    
    # Calculate button
    if st.sidebar.button("Run Simulation", type="primary"):
        # Validate inputs
        if max_bet < min_bet:
            st.error("Maximum bet must be greater than or equal to minimum bet")
            return
        
        # Initialize calculator
        calculator = BlackjackCalculator(
            num_decks=num_decks,
            min_bet=min_bet,
            max_bet=max_bet,
            starting_bankroll=starting_bankroll,
            win_rate_percent=win_rate,
            hands_per_hour=hands_per_hour
        )
        
        # Display basic calculations
        st.header("📊 Simulation Results")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_hands = hours_played * hands_per_hour
        hourly_ev = calculator.calculate_hourly_ev()
        total_ev = hourly_ev * hours_played
        risk_of_ruin = calculator.calculate_risk_of_ruin(hours_played)
        
        with col1:
            st.metric("Total Expected Value", f"${total_ev:.2f}")
        
        with col2:
            st.metric("Hourly Expected Value", f"${hourly_ev:.2f}")
        
        with col3:
            st.metric("Risk of Ruin", f"{risk_of_ruin:.2f}%")
        
        with col4:
            st.metric("Total Hands", f"{total_hands:,}")
        
        # Monte Carlo simulation
        with st.spinner("Running Monte Carlo simulation..."):
            simulation = MonteCarloSimulation(calculator)
            simulation_results = simulation.run_simulation(
                hours_played=hours_played,
                num_runs=monte_carlo_runs
            )
        
        # Visualizations
        visualizer = Visualizer()
        
        # Expected value over time
        st.subheader("Expected Value Over Time")
        ev_fig = visualizer.plot_expected_value_over_time(
            hours_played, hourly_ev, calculator.calculate_hourly_std()
        )
        st.plotly_chart(ev_fig, use_container_width=True)
        
        # Monte Carlo results
        st.subheader("Monte Carlo Simulation Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            bankroll_fig = visualizer.plot_bankroll_distribution(simulation_results)
            st.plotly_chart(bankroll_fig, use_container_width=True)
        
        with col2:
            trajectory_fig = visualizer.plot_sample_trajectories(
                simulation_results, hours_played, num_samples=10
            )
            st.plotly_chart(trajectory_fig, use_container_width=True)
        
        # Detailed statistics
        st.subheader("Detailed Statistics")
        
        final_bankrolls = simulation_results['final_bankrolls']
        profits = final_bankrolls - starting_bankroll
        
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.write("**Profit/Loss Statistics:**")
            st.write(f"- Average Profit: ${np.mean(profits):.2f}")
            st.write(f"- Median Profit: ${np.median(profits):.2f}")
            st.write(f"- Standard Deviation: ${np.std(profits):.2f}")
            st.write(f"- 95% Confidence Interval: ${np.percentile(profits, 2.5):.2f} to ${np.percentile(profits, 97.5):.2f}")
        
        with stats_col2:
            st.write("**Probability Analysis:**")
            prob_profit = (profits > 0).mean() * 100
            prob_double = (final_bankrolls >= 2 * starting_bankroll).mean() * 100
            prob_ruin = (final_bankrolls <= 0).mean() * 100
            
            st.write(f"- Probability of Profit: {prob_profit:.1f}%")
            st.write(f"- Probability of Doubling Bankroll: {prob_double:.1f}%")
            st.write(f"- Probability of Ruin (Simulation): {prob_ruin:.1f}%")
        
        # Risk analysis
        st.subheader("Risk Analysis")
        
        # Calculate additional risk metrics
        drawdown_analysis = visualizer.analyze_drawdowns(simulation_results['trajectories'])
        
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            st.write("**Drawdown Analysis:**")
            st.write(f"- Average Maximum Drawdown: ${drawdown_analysis['avg_max_drawdown']:.2f}")
            st.write(f"- 95th Percentile Drawdown: ${drawdown_analysis['p95_drawdown']:.2f}")
            st.write(f"- Worst Case Drawdown: ${drawdown_analysis['worst_drawdown']:.2f}")
        
        with risk_col2:
            st.write("**Bankroll Requirements:**")
            recommended_bankroll = calculator.calculate_recommended_bankroll()
            current_adequacy = (starting_bankroll / recommended_bankroll) * 100
            
            st.write(f"- Recommended Bankroll: ${recommended_bankroll:.2f}")
            st.write(f"- Current Adequacy: {current_adequacy:.1f}%")
            if current_adequacy < 100:
                st.warning("⚠️ Your bankroll may be insufficient for this spread")
            else:
                st.success("✅ Your bankroll appears adequate")

    # Information section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 About This Tool")
    st.sidebar.markdown("""
    This simulation helps card counters understand:
    - **Expected Value**: Long-term profit expectations
    - **Variance**: Natural fluctuations in results
    - **Risk of Ruin**: Probability of losing entire bankroll
    - **Bankroll Requirements**: Recommended starting capital
    
    **Note**: This tool is for educational purposes only.
    """)

if __name__ == "__main__":
    main()
