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
    
    # Create placeholder for live calculations
    live_stats_placeholder = st.empty()
    
    # Sidebar for inputs
    st.sidebar.header("Table Rules Configuration")
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 True Count Data")
    st.sidebar.markdown("Data will update when simulation runs")
    
    # Deck configuration
    deck_options = [1, 2, 4, 6]
    num_decks = st.sidebar.selectbox(
        "Number of Decks in Shoe",
        options=deck_options,
        index=3,
        help="Standard casinos use 6-deck shoes"
    )
    
    # Get available penetration options from CSV files
    def get_available_penetrations(num_decks):
        import os
        import glob
        
        # Look for CSV files matching the deck count
        pattern = f"true count distributions/{num_decks}decks-*.csv"
        files = glob.glob(pattern)
        
        penetrations = []
        for file in files:
            filename = os.path.basename(file)
            if "nopenetration" in filename:
                penetrations.append((num_decks, f"{num_decks} decks (100% - No penetration)"))
            else:
                # Extract penetration value from filename like "1decks-0.75penetration.csv"
                pen_part = filename.split('-')[1].replace('penetration.csv', '')
                try:
                    pen_value = float(pen_part)
                    percentage = (pen_value / num_decks) * 100
                    penetrations.append((pen_value, f"{pen_value} decks ({percentage:.1f}%)"))
                except ValueError:
                    continue
        
        # Sort by penetration value (descending for best first)
        penetrations.sort(key=lambda x: x[0], reverse=True)
        return penetrations
    
    penetration_options = get_available_penetrations(num_decks)
    
    if penetration_options:
        penetration_deck, penetration_label = st.sidebar.selectbox(
            "Deck Penetration",
            options=penetration_options,
            index=0,  # Default to best penetration
            format_func=lambda x: x[1],
            help="How many decks are dealt before shuffle (based on available simulation data)"
        )
    else:
        st.sidebar.error(f"No penetration data available for {num_decks} decks")
        penetration_deck = num_decks
    
    # Dealer rules
    dealer_hits_soft17 = st.sidebar.selectbox(
        "Dealer Hits Soft 17",
        options=[True, False],
        index=0,
        format_func=lambda x: "Yes" if x else "No",
        help="Does dealer hit or stand on soft 17"
    )
    
    # Doubling rules
    double_after_split = st.sidebar.selectbox(
        "Double After Split (DAS)",
        options=[True, False],
        index=0,
        format_func=lambda x: "Yes" if x else "No",
        help="Can you double down after splitting"
    )
    
    # Splitting rules
    can_split_aces = st.sidebar.selectbox(
        "Split Aces Allowed",
        options=[True, False],
        index=0,
        format_func=lambda x: "Yes" if x else "No",
        help="Can you split aces"
    )
    
    resplit_aces = False
    if can_split_aces:
        resplit_aces = st.sidebar.selectbox(
            "Resplit Aces",
            options=[True, False],
            index=1,
            format_func=lambda x: "Yes" if x else "No",
            help="Can you resplit aces if you get another ace"
        )
    
    max_splits = st.sidebar.selectbox(
        "Maximum Split Hands",
        options=[2, 3, 4],
        index=1,
        help="Maximum number of hands after splitting"
    )
    
    # Surrender
    surrender_allowed = st.sidebar.selectbox(
        "Late Surrender",
        options=[True, False],
        index=1,
        format_func=lambda x: "Yes" if x else "No",
        help="Can you surrender after dealer checks for blackjack"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("Simulation Parameters")
    
    # Betting Strategy Configuration
    st.sidebar.subheader("Betting Strategy ($)")
    st.sidebar.write("Set your bet amount for each true count:")
    
    # Initialize session state for betting strategy with TC -3 to +6
    if 'bet_amounts' not in st.session_state:
        st.session_state.bet_amounts = {
            -3: 0, -2: 0, -1: 0, 0: 0, 1: 5,
            2: 10, 3: 15, 4: 25, 5: 25, 6: 25
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
        value=5000,
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
    
    num_runs = st.sidebar.number_input(
        "Number of Simulation Runs",
        min_value=1000,
        max_value=50000,
        value=5000,
        step=1000,
        help="More runs provide better statistical accuracy but take longer to compute"
    )
    


    # Calculate button
    if st.sidebar.button("Run Simulation", type="primary"):
        # Validate betting strategy
        if len(st.session_state.betting_strategy) < 2:
            st.error("Please define at least 2 betting levels")
            return
        
        # Prepare table rules for calculator
        table_rules = {
            'penetration_deck': penetration_deck,
            'dealer_hits_soft_17': dealer_hits_soft17,
            'double_after_split': double_after_split,
            'can_split_aces': can_split_aces,
            'resplit_aces': resplit_aces,
            'max_splits': max_splits,
            'surrender_allowed': surrender_allowed
        }
        
        # Initialize calculator
        calculator = BlackjackCalculator(
            num_decks=num_decks,
            starting_bankroll=starting_bankroll,
            hands_per_hour=hands_per_hour,
            betting_strategy=st.session_state.betting_strategy,
            table_rules=table_rules
        )
        
        # Display table rules summary in tabs
        st.header("🎲 Table Rules Configuration")
        
        # Create tabs for table rules organization
        rules_tab1, rules_tab2, rules_tab3 = st.tabs(["🎰 Game Setup", "👤 Player Options", "📊 Data Source"])
        
        with rules_tab1:
            st.subheader("Basic Game Configuration")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Decks", f"{num_decks}")
                st.metric("Penetration", f"{penetration_deck} decks ({(penetration_deck/num_decks)*100:.1f}%)")
            with col2:
                st.metric("Dealer on Soft 17", "Hits" if dealer_hits_soft17 else "Stands")
                st.metric("Hands per Hour", f"{hands_per_hour}")
        
        with rules_tab2:
            st.subheader("Available Player Actions")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"✅ **Double after split:** {'Yes' if double_after_split else 'No'}")
                st.write(f"✅ **Split aces:** {'Yes' if can_split_aces else 'No'}")
                if can_split_aces:
                    st.write(f"✅ **Resplit aces:** {'Yes' if resplit_aces else 'No'}")
            with col2:
                st.write(f"✅ **Max split hands:** {max_splits}")
                st.write(f"✅ **Late surrender:** {'Yes' if surrender_allowed else 'No'}")
        
        with rules_tab3:
            st.subheader("Simulation Data Source")
            # Show which simulation data is being used
            if penetration_deck == num_decks:
                csv_file = f"{num_decks}decks-nopenetration.csv"
            else:
                csv_file = f"{num_decks}decks-{penetration_deck}penetration.csv"
            
            st.write(f"**File:** {csv_file}")
            st.write("**Source:** Real simulation data from 1M+ shoes")
            st.write("**Method:** Professional blackjack simulation using High-Low counting system")
            st.info("💡 This data represents actual true count frequencies from comprehensive blackjack simulations, ensuring accurate edge calculations.")
        
        # Display calculation results
        st.header("📊 Simulation Results")
        
        # Show calculated edge and betting strategy summary
        st.info(f"**Calculated Edge**: {calculator.edge*100:.3f}% (weighted by bet size and count frequencies)")
        


        # Show CSV data and calculations
        with st.expander("📊 CSV Data & Edge Breakdown"):
            # Show which CSV file is being used
            if penetration_deck == num_decks:
                csv_filename = f"{num_decks}decks-nopenetration.csv"
            else:
                csv_filename = f"{num_decks}decks-{penetration_deck}penetration.csv"
            
            st.write(f"**Data Source:** {csv_filename}")
            st.write("**Raw CSV True Count Frequencies:**")
            
            # Display raw frequencies from CSV
            csv_data = []
            for tc in range(-3, 7):
                frequency = calculator.count_frequencies.get(tc, 0)
                csv_data.append(f"TC {tc:+d}: {frequency*100:.2f}%")
            
            # Show in columns for better layout
            col1, col2 = st.columns(2)
            with col1:
                for i in range(0, 5):
                    st.write(csv_data[i])
            with col2:
                for i in range(5, 10):
                    st.write(csv_data[i])
            
            st.write("**Edge Calculations Using CSV Data:**")
            total_weighted_edge = 0
            total_weighted_bet = 0
            
            for tc in range(-3, 7):
                frequency = calculator.count_frequencies.get(tc, 0)
                edge = calculator.count_edges.get(tc, 0)
                bet_amount = calculator._get_bet_for_count(tc)
                contribution = frequency * edge * bet_amount
                
                total_weighted_edge += contribution
                total_weighted_bet += frequency * bet_amount
                
                st.write(f"TC {tc:+d}: {frequency*100:.2f}% freq × {edge*100:.1f}% edge × ${bet_amount} bet = {contribution*100:.3f}% contribution")
            
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
            hours_played, hourly_ev, calculator.calculate_hourly_std(), starting_bankroll
        )
        st.plotly_chart(ev_fig, use_container_width=True)
        
        # Run comprehensive Monte Carlo analysis
        st.subheader(f"Monte Carlo Analysis ({num_runs:,} Runs)")
        
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
        st.subheader(f"Statistical Analysis ({num_runs:,} Simulations)")
        
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
        
        # Create tabbed interface for simulation analysis
        tab1, tab2, tab3 = st.tabs(["📊 Distribution & Performance", "⚠️ Risk Analysis", "📈 Advanced Metrics"])
        
        with tab1:
            # Histogram of final bankrolls
            st.subheader("Distribution of Final Bankrolls")
            bankroll_fig = visualizer.plot_bankroll_distribution(monte_carlo_results)
            st.plotly_chart(bankroll_fig, use_container_width=True)
        
        with tab2:
            # Risk of Ruin for different bankrolls
            st.subheader("Risk of Ruin Analysis")
            st.write("Shows probability of losing your entire bankroll at different starting amounts:")
            
            # Calculate RoR based on simulation results
            ror_data = []
            bankroll_percentages = [50, 75, 100, 125, 150]
            current_bankroll = starting_bankroll
            
            for pct in bankroll_percentages:
                test_bankroll = current_bankroll * pct / 100
                # Scale the simulation results proportionally to the test bankroll
                scale_factor = test_bankroll / current_bankroll
                scaled_final_bankrolls = monte_carlo_results['final_bankrolls'] * scale_factor
                ruin_count = np.sum(scaled_final_bankrolls <= 0)
                ror_percentage = (ruin_count / len(scaled_final_bankrolls)) * 100
                ror_data.append({
                    "Bankroll Size": f"{pct}% of current (${test_bankroll:,.0f})", 
                    "Risk of Ruin": f"{ror_percentage:.1f}%"
                })
            
            # Display RoR table
            ror_df = pd.DataFrame(ror_data)
            st.table(ror_df)
            
            # Add interpretation
            st.info("💡 **Interpretation:** Lower percentages indicate safer bankroll levels. Professional players typically aim for risk of ruin below 5-10%.")
        
        with tab3:
            # Calculate N₀ (hands to be 1 SD above breakeven)
            st.subheader("N₀ Analysis (Hands to 1 SD Above Breakeven)")
            
            ev_per_hand = calculator.calculate_hourly_ev() / calculator.hands_per_hour
            std_per_hand = calculator.calculate_hourly_std() / np.sqrt(calculator.hands_per_hour)
            variance_per_hand = std_per_hand ** 2
            
            if ev_per_hand > 0 and variance_per_hand > 0:
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
        
        # Use the pre-calculated average trajectory from Monte Carlo results
        avg_trajectory = monte_carlo_results.get('avg_trajectory', [])
        
        if avg_trajectory:
            avg_profit = avg_trajectory[-1] - starting_bankroll
        else:
            avg_profit = 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Expected Profit", f"${total_ev:.2f}")
        with col2:
            st.metric("Average Actual Profit", f"${avg_profit:.2f}", 
                     delta=f"${avg_profit - total_ev:.2f}")
        
        # First graph: Expected vs Average Actual
        trajectory_fig = visualizer.plot_average_trajectory_vs_expected(
            avg_trajectory, hours_played, hourly_ev, starting_bankroll
        )
        st.plotly_chart(trajectory_fig, use_container_width=True)
        
        # Second graph: Best vs Worst Case Analysis
        st.subheader("Best vs Worst Case Scenarios")
        
        # Find best and worst case trajectories
        final_profits = [traj[-1] - starting_bankroll for traj in monte_carlo_results['trajectories']]
        best_idx = np.argmax(final_profits)
        worst_idx = np.argmin(final_profits)
        best_trajectory = monte_carlo_results['trajectories'][best_idx]
        worst_trajectory = monte_carlo_results['trajectories'][worst_idx]
        
        # Show best/worst profit metrics
        best_profit = best_trajectory[-1] - starting_bankroll
        worst_profit = worst_trajectory[-1] - starting_bankroll
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best Case Profit", f"${best_profit:.2f}", 
                     delta=f"${best_profit - total_ev:.2f} vs Expected")
        with col2:
            st.metric("Worst Case Profit", f"${worst_profit:.2f}", 
                     delta=f"${worst_profit - total_ev:.2f} vs Expected")
        
        # Plot best vs worst trajectories
        extremes_fig = visualizer.plot_best_vs_worst_trajectories(
            best_trajectory, worst_trajectory, starting_bankroll
        )
        st.plotly_chart(extremes_fig, use_container_width=True)
        
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
            recommended_bankroll = calculator.calculate_recommended_bankroll()
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
        


    # Calculate live statistics if we have valid inputs
    if penetration_options and len(st.session_state.betting_strategy) >= 2:
        try:
            # Create temporary calculator for live display
            temp_table_rules = {
                'penetration_deck': penetration_deck,
                'dealer_hits_soft_17': dealer_hits_soft17,
                'double_after_split': double_after_split,
                'can_split_aces': can_split_aces,
                'resplit_aces': resplit_aces,
                'max_splits': max_splits,
                'surrender_allowed': surrender_allowed
            }
            
            temp_calculator = BlackjackCalculator(
                num_decks=num_decks,
                starting_bankroll=starting_bankroll,
                hands_per_hour=hands_per_hour,
                betting_strategy=st.session_state.betting_strategy,
                table_rules=temp_table_rules
            )
            
            # Calculate key metrics using current sidebar values
            hourly_ev = temp_calculator.calculate_hourly_ev()
            hourly_std = temp_calculator.calculate_hourly_std()
            
            # Use the actual hours_played from sidebar
            risk_of_ruin = temp_calculator.calculate_risk_of_ruin(hours_played)
            
            # Calculate hours to break even (when EV = starting bankroll)
            if hourly_ev > 0:
                hours_to_breakeven = starting_bankroll / hourly_ev
            else:
                hours_to_breakeven = float('inf')
            
            # Display live statistics
            with live_stats_placeholder.container():
                st.markdown("### 📊 Live Strategy Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Expected Value/Hour",
                        value=f"${hourly_ev:.2f}",
                        delta=f"{temp_calculator.edge*100:.2f}% edge"
                    )
                
                with col2:
                    st.metric(
                        label="Standard Deviation/Hour", 
                        value=f"${hourly_std:.2f}",
                        delta=f"±{hourly_std/hourly_ev:.1f}x EV" if hourly_ev > 0 else None
                    )
                
                with col3:
                    if risk_of_ruin < 100:
                        color = "🟢" if risk_of_ruin < 5 else "🟡" if risk_of_ruin < 15 else "🔴"
                        st.metric(
                            label="Risk of Ruin",
                            value=f"{risk_of_ruin:.1f}%",
                            delta=f"{color} {hours_played}h play"
                        )
                    else:
                        st.metric(
                            label="Risk of Ruin",
                            value="High",
                            delta="⚠️ Reconsider strategy"
                        )
                
                with col4:
                    if hours_to_breakeven != float('inf'):
                        if hours_to_breakeven <= 8760:  # 1 year
                            st.metric(
                                label="Hours to 2x Bankroll",
                                value=f"{hours_to_breakeven:.0f}h",
                                delta=f"{hours_to_breakeven/24:.1f} days"
                            )
                        else:
                            st.metric(
                                label="Hours to 2x Bankroll", 
                                value=">1 year",
                                delta="💰 Long term"
                            )
                    else:
                        st.metric(
                            label="Hours to 2x Bankroll",
                            value="Never",
                            delta="📉 Negative EV"
                        )
                
                # Show data source and debug info
                if penetration_deck == num_decks:
                    csv_filename = f"{num_decks}decks-nopenetration.csv"
                else:
                    csv_filename = f"{num_decks}decks-{penetration_deck}penetration.csv"
                
                # Debug: Show detailed calculation breakdown
                ev_per_hand = temp_calculator._calculate_ev_per_hand()
                avg_bet = temp_calculator.avg_bet
                edge = temp_calculator.edge
                
                # Calculate high positive count contribution
                import os, csv
                penetration_deck = temp_calculator.table_rules['penetration_deck']
                if penetration_deck == num_decks:
                    debug_filename = f"true count distributions/{num_decks}decks-nopenetration.csv"
                else:
                    debug_filename = f"true count distributions/{num_decks}decks-{penetration_deck}penetration.csv"
                
                high_pos_freq = 0
                high_pos_ev_contrib = 0
                
                if os.path.exists(debug_filename):
                    with open(debug_filename, 'r') as file:
                        reader = csv.reader(file)
                        for row in reader:
                            if row and not row[0].startswith('#') and row[0] != 'True Count':
                                try:
                                    tc = int(row[0])
                                    percentage = float(row[1])
                                    if tc >= 6:  # High positive counts
                                        freq = percentage / 100.0
                                        bet = temp_calculator._get_bet_for_count(tc)
                                        edge_val = temp_calculator._get_actual_edge_for_count(tc)
                                        high_pos_freq += freq
                                        high_pos_ev_contrib += edge_val * bet * freq
                                except (ValueError, IndexError):
                                    continue
                
                st.caption(f"📁 Data: {csv_filename}")
                st.caption(f"🔍 EV/hand: ${ev_per_hand:.4f} | High TC6+: {high_pos_freq*100:.2f} percent | High EV contrib: ${high_pos_ev_contrib:.4f}")
        
        except Exception as e:
            with live_stats_placeholder.container():
                st.info("💡 Configure your betting strategy to see live analysis")
    else:
        with live_stats_placeholder.container():
            st.info("💡 Complete your betting strategy to see live analysis")

    # Information section - will be updated after calculator is created  
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 📖 About This Tool")
    st.sidebar.markdown("""
    Set your betting strategy for each true count from -3 to +6. The tool calculates your expected value based on realistic count frequencies and edge estimates.
    
    **Note**: This tool is for educational purposes only.
    """)

if __name__ == "__main__":
    main()
