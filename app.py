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
    st.sidebar.subheader("Betting Strategy ($)")
    st.sidebar.write("Set your bet amount for each true count:")
    
    # Initialize session state for betting strategy with TC -3 to +6
    if 'bet_amounts' not in st.session_state:
        st.session_state.bet_amounts = {
            -3: 10, -2: 10, -1: 10, 0: 10, 1: 10,
            2: 50, 3: 100, 4: 150, 5: 200, 6: 250
        }
    
    # Create betting strategy inputs for each true count
    betting_strategy = []
    for tc in range(-3, 7):  # TC from -3 to +6
        bet_amount = st.sidebar.number_input(
            f"TC {tc:+d}",
            min_value=0,
            max_value=10000,
            value=st.session_state.bet_amounts[tc],
            step=5,
            key=f"bet_tc_{tc}",
            help=f"Bet amount ($) when true count is {tc} (0 = sit out)"
        )
        st.session_state.bet_amounts[tc] = bet_amount
        betting_strategy.append({'true_count': tc, 'bet_amount': bet_amount})
    
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
        
        # Show current game rules
        with st.expander("🎲 Game Rules (OLG Blackjack)"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Current Rule Set:**")
                st.write(f"- Decks: {calculator.game_rules['decks']}")
                st.write(f"- Dealer hits soft 17: {'Yes' if calculator.game_rules['dealer_hits_soft_17'] else 'No'}")
                st.write(f"- Blackjack payout: {calculator.game_rules['blackjack_payout']}:1")
                st.write(f"- Double any two cards: {'Yes' if calculator.game_rules['double_any_two'] else 'No'}")
                st.write(f"- Double after split: {'Yes' if calculator.game_rules['double_after_split'] else 'No'}")
            with col2:
                st.write("**Additional Rules:**")
                st.write(f"- Resplit aces: {'Yes' if calculator.game_rules['resplit_aces'] else 'No'}")
                st.write(f"- Late surrender: {'Yes' if calculator.game_rules['late_surrender'] else 'No'}")
                st.write(f"- European no hole card: {'Yes' if calculator.game_rules['european_no_hole_card'] else 'No'}")
                st.write(f"- Insurance offered: {'Yes' if calculator.game_rules['insurance_offered'] else 'No'}")
                st.write(f"- **Base house edge: {calculator.base_house_edge*100:.1f}%**")

        # Debug: Show detailed edge breakdown
        with st.expander("📊 Edge Calculation Breakdown"):
            st.write("**Count Frequencies and Contributions:**")
            total_weighted_edge = 0
            total_weighted_bet = 0
            
            for tc in range(-3, 7):
                frequency = calculator.count_frequencies.get(tc, 0)
                edge = calculator.count_edges.get(tc, 0)
                bet_amount = calculator._get_bet_for_count(tc)
                contribution = frequency * edge * bet_amount
                
                total_weighted_edge += contribution
                total_weighted_bet += frequency * bet_amount
                
                st.write(f"TC {tc:+d}: {frequency*100:.1f}% frequency, {edge*100:.1f}% edge, ${bet_amount} bet → {contribution*100:.3f}% contribution")
            
            st.write(f"**Total weighted edge: {total_weighted_edge:.6f}**")
            st.write(f"**Total weighted bet: {total_weighted_bet:.2f}**")
            st.write(f"**Final edge: {(total_weighted_edge/total_weighted_bet)*100:.3f}%**")
            st.write(f"**Average bet: ${calculator.avg_bet:.2f}**")
        
        # Show betting strategy
        strategy_text = "**Your Betting Strategy**: "
        for rule in sorted(st.session_state.betting_strategy, key=lambda x: x['true_count']):
            if rule['bet_amount'] == 0:
                strategy_text += f"TC {rule['true_count']}+ → sit out, "
            else:
                strategy_text += f"TC {rule['true_count']}+ → ${rule['bet_amount']}, "
        st.write(strategy_text.rstrip(", "))
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_hands = hours_played * hands_per_hour
        hourly_ev = calculator.calculate_hourly_ev()
        total_ev = hourly_ev * hours_played
        risk_of_ruin = calculator.calculate_risk_of_ruin(hours_played)
        
        with col1:
            st.metric("Hourly Expected Value", f"${hourly_ev:.2f}")
        
        with col2:
            st.metric("Total Expected Value", f"${total_ev:.2f}")
        
        with col3:
            st.metric("Risk of Ruin", f"{risk_of_ruin:.2f}%")
        
        with col4:
            st.metric("Total Hands", f"{total_hands:,}")
        
        # Visualizations
        visualizer = Visualizer()
        
        # Expected value over time
        st.subheader("Expected Value Over Time")
        ev_fig = visualizer.plot_expected_value_over_time(
            hours_played, hourly_ev, calculator.calculate_hourly_std()
        )
        st.plotly_chart(ev_fig, use_container_width=True)
        
        # Run comprehensive Monte Carlo analysis
        st.subheader("Monte Carlo Analysis (10,000 Runs)")
        
        # Create progress bar with text
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Starting simulation... 0% complete")
        
        # Custom progress callback to update both bar and text
        def update_progress(current, total):
            percentage = current / total
            progress_bar.progress(percentage)
            status_text.text(f"Running simulation... {percentage*100:.0f}% complete ({current:,} / {total:,} runs)")
        
        simulation = MonteCarloSimulation(calculator)
        
        # Modified simulation call with custom progress updates
        results = []
        num_runs = 10000
        
        for i in range(num_runs):
            result = simulation.run_single_simulation(hours_played)
            results.append(result)
            
            # Update progress every 50 runs
            if (i + 1) % 50 == 0 or i == num_runs - 1:
                update_progress(i + 1, num_runs)
        
        # Process results like the original run_simulation method
        final_bankrolls = [r['final_bankroll'] for r in results]
        trajectories = [r['trajectory'] for r in results]
        max_bankrolls = [r['max_bankroll'] for r in results]
        min_bankrolls = [r['min_bankroll'] for r in results]
        actual_edges = [r['actual_avg_edge'] for r in results]
        
        profits = [fb - starting_bankroll for fb in final_bankrolls]
        
        monte_carlo_results = {
            'final_bankrolls': np.array(final_bankrolls),
            'trajectories': trajectories,
            'profits': np.array(profits),
            'max_bankrolls': np.array(max_bankrolls),
            'min_bankrolls': np.array(min_bankrolls),
            'actual_edges': np.array(actual_edges),
            'statistics': {
                'mean_profit': np.mean(profits),
                'median_profit': np.median(profits),
                'std_profit': np.std(profits),
                'min_profit': np.min(profits),
                'max_profit': np.max(profits),
                'prob_profit': np.mean(np.array(profits) > 0),
                'prob_ruin': np.mean(np.array(final_bankrolls) <= 0),
                'prob_double': np.mean(np.array(final_bankrolls) >= 2 * starting_bankroll),
                'mean_actual_edge': np.mean(actual_edges),
                'std_actual_edge': np.std(actual_edges)
            }
        }
        
        single_result = simulation.run_single_simulation(hours_played)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.text("✅ Simulation complete!")
        
        # Monte Carlo Results Analysis
        final_bankrolls = monte_carlo_results['final_bankrolls']
        profits = monte_carlo_results['profits']
        
        # Statistical Analysis
        st.subheader("Statistical Analysis (10,000 Simulations)")
        
        # Display key statistics
        mean_profit = np.mean(profits)
        std_profit = np.std(profits)
        confidence_interval_lower = mean_profit - 1.96 * std_profit
        confidence_interval_upper = mean_profit + 1.96 * std_profit
        
        # Display actual vs theoretical edge
        theoretical_edge = calculator.edge
        actual_edge = monte_carlo_results['statistics']['mean_actual_edge']
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Mean Profit", f"${mean_profit:.2f}")
        with stat_col2:
            st.metric("Standard Deviation", f"${std_profit:.2f}")
        with stat_col3:
            st.metric("Theoretical Edge", f"{theoretical_edge*100:.3f}%")
        with stat_col4:
            st.metric("Actual Avg Edge", f"{actual_edge*100:.3f}%", 
                     delta=f"{(actual_edge-theoretical_edge)*100:.3f}%")
        
        # Second row with confidence interval
        st.metric("95% Confidence Interval", f"${confidence_interval_lower:.0f} to ${confidence_interval_upper:.0f}")
        
        # Histogram of final bankrolls
        st.subheader("Distribution of Final Bankrolls")
        bankroll_fig = visualizer.plot_bankroll_distribution(monte_carlo_results)
        st.plotly_chart(bankroll_fig, use_container_width=True)
        
        # Risk of Ruin for different bankrolls
        st.subheader("Risk of Ruin Analysis")
        
        # Calculate RoR for different bankroll percentages
        ror_data = []
        bankroll_percentages = [50, 75, 100, 125, 150]
        recommended_bankroll = calculator.calculate_recommended_bankroll()
        
        for pct in bankroll_percentages:
            test_bankroll = (recommended_bankroll * pct / 100) if recommended_bankroll != float('inf') else starting_bankroll * pct / 100
            # Create temporary calculator with different bankroll
            temp_calculator = BlackjackCalculator(
                num_decks=num_decks,
                starting_bankroll=test_bankroll,
                hands_per_hour=hands_per_hour,
                betting_strategy=st.session_state.betting_strategy
            )
            temp_ror = temp_calculator.calculate_risk_of_ruin(hours_played)
            ror_data.append({"Bankroll Size": f"{pct}% of recommended", "Risk of Ruin": f"{temp_ror:.1f}%"})
        
        # Display RoR table
        ror_df = pd.DataFrame(ror_data)
        st.table(ror_df)
        
        # Calculate N₀ (hands to be 1 SD above breakeven)
        st.subheader("N₀ Analysis (Hands to 1 SD Above Breakeven)")
        
        ev_per_hand = calculator.calculate_hourly_ev() / calculator.hands_per_hour
        variance_per_hand = (calculator.calculate_hourly_std() / calculator.hands_per_hour) ** 2
        
        if ev_per_hand > 0:
            n0_hands = variance_per_hand / (ev_per_hand ** 2)
            n0_hours = n0_hands / calculator.hands_per_hour
            
            n0_col1, n0_col2 = st.columns(2)
            with n0_col1:
                st.metric("N₀ (Hands)", f"{n0_hands:,.0f}")
            with n0_col2:
                st.metric("N₀ (Hours)", f"{n0_hours:.1f}")
            
            st.info(f"After {n0_hands:,.0f} hands ({n0_hours:.1f} hours), your expected profit will equal one standard deviation of variance. This is when skill begins to significantly outweigh luck.")
        else:
            st.error("Negative edge detected - N₀ calculation not applicable")
        
        # Average trajectory analysis
        st.subheader("Average Performance vs Expected")
        
        # Calculate average trajectory across all simulations
        max_hours = len(monte_carlo_results['trajectories'][0]) - 1
        avg_trajectory = []
        
        for hour in range(max_hours + 1):
            hour_values = []
            for trajectory in monte_carlo_results['trajectories']:
                if hour < len(trajectory):
                    hour_values.append(trajectory[hour])
                else:
                    hour_values.append(0)  # Ruined simulations
            avg_trajectory.append(np.mean(hour_values))
        
        avg_profit = avg_trajectory[-1] - starting_bankroll
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Expected Profit", f"${total_ev:.2f}")
        with col2:
            st.metric("Average Actual Profit", f"${avg_profit:.2f}", 
                     delta=f"${avg_profit - total_ev:.2f}")
        
        # Plot average trajectory vs expected
        trajectory_fig = visualizer.plot_average_trajectory_vs_expected(
            avg_trajectory, hours_played, hourly_ev, starting_bankroll
        )
        st.plotly_chart(trajectory_fig, use_container_width=True)
        
        # Additional risk analysis
        st.subheader("Bankroll Recommendations")
        
        risk_col1, risk_col2 = st.columns(2)
        with risk_col1:
            st.write("**Current Strategy Risk:**")
            st.write(f"- Risk of Ruin: {risk_of_ruin:.2f}%")
            if risk_of_ruin > 10:
                st.warning("⚠️ High risk - consider larger bankroll or smaller bets")
            elif risk_of_ruin > 5:
                st.warning("⚠️ Moderate risk - monitor bankroll carefully")
            else:
                st.success("✅ Low risk of ruin")
        
        with risk_col2:
            st.write("**Bankroll Adequacy:**")
            if recommended_bankroll != float('inf'):
                bankroll_adequacy = (starting_bankroll / recommended_bankroll) * 100
                st.write(f"- Recommended: ${recommended_bankroll:.2f}")
                st.write(f"- Current Adequacy: {bankroll_adequacy:.1f}%")
                if bankroll_adequacy < 100:
                    st.warning("⚠️ Consider increasing bankroll")
                else:
                    st.success("✅ Bankroll appears adequate")
            else:
                st.error("⚠️ Negative edge - not recommended")
        
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
