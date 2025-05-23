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
    

    
    hands_per_hour = st.sidebar.number_input(
        "Hands per Hour",
        min_value=50,
        max_value=500,
        value=100,
        step=10,
        help="100 for solo play, 250+ for team play"
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
            hands_per_hour=hands_per_hour
        )
        
        # Display calculation results
        st.header("📊 Simulation Results")
        
        # Show calculated edge based on count frequencies
        st.info(f"**Calculated Edge**: {calculator.edge*100:.3f}% (based on true count frequencies)")
        
        # Basic statistics
        col1, col2, col3 = st.columns(3)
        
        total_hands = hours_played * hands_per_hour
        hourly_ev = calculator.calculate_hourly_ev()
        total_ev = hourly_ev * hours_played
        
        with col1:
            st.metric("Hourly Expected Value", f"${hourly_ev:.2f}")
        
        with col2:
            st.metric("Total Expected Value", f"${total_ev:.2f}")
        
        with col3:
            st.metric("Total Hands", f"{total_hands:,}")
        
        # Visualizations
        visualizer = Visualizer()
        
        # Expected value over time
        st.subheader("Expected Value Over Time")
        ev_fig = visualizer.plot_expected_value_over_time(
            hours_played, hourly_ev, calculator.calculate_hourly_std()
        )
        st.plotly_chart(ev_fig, use_container_width=True)
        
        # Run a single randomized simulation
        st.subheader("Randomized Simulation (Single Run)")
        with st.spinner("Running randomized simulation..."):
            simulation = MonteCarloSimulation(calculator)
            single_result = simulation.run_single_simulation(hours_played)
        
        # Display actual vs expected results
        actual_profit = single_result['final_bankroll'] - starting_bankroll
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Expected Profit", f"${total_ev:.2f}")
        with col2:
            st.metric("Actual Profit (Random)", f"${actual_profit:.2f}", 
                     delta=f"${actual_profit - total_ev:.2f}")
        
        # Plot the actual trajectory vs expected
        trajectory_fig = visualizer.plot_single_trajectory_vs_expected(
            single_result['trajectory'], hours_played, hourly_ev, starting_bankroll
        )
        st.plotly_chart(trajectory_fig, use_container_width=True)

    # Information section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 True Count Frequencies")
    st.sidebar.markdown("""
    **Automatic Edge Calculation Based On:**
    - TC ≤0: 60% frequency, -0.5% edge, min bet
    - TC +1: 20% frequency, 0% edge, min bet  
    - TC +2: 10% frequency, +0.5% edge, max bet
    - TC +3: 7% frequency, +1.0% edge, max bet
    - TC +4+: 3% frequency, +1.5% edge, max bet
    """)
    
    st.sidebar.markdown("### 📖 About This Tool")
    st.sidebar.markdown("""
    This simulation shows expected value over time and compares it with a randomized single simulation to demonstrate variance.
    
    **Note**: This tool is for educational purposes only.
    """)

if __name__ == "__main__":
    main()
