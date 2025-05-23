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
    
    # Betting Strategy Configuration
    st.sidebar.subheader("Betting Strategy (Units)")
    st.sidebar.write("Set your bet for each true count:")
    
    # Initialize session state for betting strategy with TC -3 to +6
    if 'bet_units' not in st.session_state:
        st.session_state.bet_units = {
            -3: 1, -2: 1, -1: 1, 0: 1, 1: 1,
            2: 5, 3: 10, 4: 15, 5: 20, 6: 25
        }
    
    # Create betting strategy inputs for each true count
    betting_strategy = []
    for tc in range(-3, 7):  # TC from -3 to +6
        bet_units = st.sidebar.number_input(
            f"TC {tc:+d}",
            min_value=0,
            max_value=100,
            value=st.session_state.bet_units[tc],
            step=1,
            key=f"bet_tc_{tc}",
            help=f"Bet units when true count is {tc} (0 = sit out)"
        )
        st.session_state.bet_units[tc] = bet_units
        betting_strategy.append({'true_count': tc, 'bet_units': bet_units})
    
    # Store the betting strategy for the calculator
    st.session_state.betting_strategy = betting_strategy
    
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
        # Validate betting strategy
        if len(st.session_state.betting_strategy) < 2:
            st.error("Please define at least 2 betting levels")
            return
        
        # Initialize calculator
        calculator = BlackjackCalculator(
            num_decks=num_decks,
            starting_bankroll=starting_bankroll,
            hands_per_hour=hands_per_hour,
            betting_strategy=st.session_state.betting_strategy
        )
        
        # Display calculation results
        st.header("📊 Simulation Results")
        
        # Show calculated edge and betting strategy summary
        st.info(f"**Calculated Edge**: {calculator.edge*100:.3f}% (weighted by bet size and count frequencies)")
        
        # Show betting strategy
        strategy_text = "**Your Betting Strategy**: "
        for rule in sorted(st.session_state.betting_strategy, key=lambda x: x['true_count']):
            strategy_text += f"TC {rule['true_count']}+ → {rule['bet_units']} units, "
        st.write(strategy_text.rstrip(", "))
        
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
    st.sidebar.markdown("### 📊 True Count Edge Calculation")
    st.sidebar.markdown("""
    **Edge per True Count:**
    - Negative TC: -0.5% + (TC × 0.1%)
    - Positive TC: TC × 0.5%
    
    **Frequencies Used:**
    - TC -3: 15%, TC -2: 18%, TC -1: 22%
    - TC 0: 20%, TC +1: 12%, TC +2: 8%
    - TC +3: 3%, TC +4: 1.5%, TC +5: 0.5%, TC +6: 0.25%
    """)
    
    st.sidebar.markdown("### 📖 About This Tool")
    st.sidebar.markdown("""
    Set your betting strategy for each true count from -3 to +6. The tool calculates your expected value based on realistic count frequencies and edge estimates.
    
    **Note**: This tool is for educational purposes only.
    """)

if __name__ == "__main__":
    main()
