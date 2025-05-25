import numpy as np
from typing import Dict, List
import concurrent.futures
from blackjack_calculator import BlackjackCalculator

class MonteCarloSimulation:
    """
    Monte Carlo simulation engine for blackjack card counting analysis.
    Runs multiple simulations to analyze variance and risk.
    """
    
    def __init__(self, calculator: BlackjackCalculator):
        self.calculator = calculator
    
    def run_single_simulation(self, hours_played: int) -> Dict:
        """
        Run a single simulation for the specified number of hours.
        Returns trajectory and final results.
        """
        total_hands = hours_played * self.calculator.hands_per_hour
        bankroll = self.calculator.starting_bankroll
        
        # Track bankroll over time (hourly snapshots)
        bankroll_trajectory = [bankroll]
        hands_played = 0
        
        # Track actual edge experienced during simulation
        total_edge_weighted = 0
        total_bet_amount = 0
        
        for hour in range(hours_played):
            # Simulate hands for this hour
            for _ in range(self.calculator.hands_per_hour):
                if bankroll <= 0:
                    # Ruin occurred
                    break
                
                # Get random true count for this hand
                rand = np.random.random()
                cumulative_freq = 0
                selected_tc = None
                
                for true_count in sorted(self.calculator.count_frequencies.keys()):
                    cumulative_freq += self.calculator.count_frequencies[true_count]
                    if rand <= cumulative_freq:
                        selected_tc = true_count
                        break
                
                if selected_tc is None:
                    selected_tc = 0
                
                # Get edge and bet for this count
                edge = self.calculator.count_edges[selected_tc]
                bet_amount = self.calculator._get_bet_for_count(selected_tc)
                
                # Skip hands where bet is 0 (sitting out)
                if bet_amount == 0:
                    hands_played += 1
                    continue
                
                # Track actual edge experienced (only for hands where we bet)
                total_edge_weighted += edge * bet_amount
                total_bet_amount += bet_amount
                
                # Simulate hand outcome
                hand_result = np.random.normal(
                    loc=edge * bet_amount,
                    scale=self.calculator.std_dev_per_hand * bet_amount
                )
                bankroll += hand_result
                hands_played += 1
            
            # Check for bankruptcy first
            if bankroll <= 0:
                # Record bankruptcy and fill remaining hours with 0
                bankroll_trajectory.append(0)
                bankroll_trajectory.extend([0] * (hours_played - hour - 1))
                bankroll = 0  # Set to 0 for final_bankroll calculation
                break
            else:
                # Record bankroll at end of hour
                bankroll_trajectory.append(bankroll)
        
        # Calculate actual average edge experienced
        actual_avg_edge = (total_edge_weighted / total_bet_amount) if total_bet_amount > 0 else 0
        
        return {
            'final_bankroll': max(0, bankroll),
            'trajectory': bankroll_trajectory,
            'hands_played': hands_played,
            'max_bankroll': max(bankroll_trajectory),
            'min_bankroll': min(bankroll_trajectory),
            'actual_avg_edge': actual_avg_edge,
            'total_bet_amount': total_bet_amount
        }
    
    def run_simulation(self, hours_played: int, num_runs: int = 1000, progress_bar=None) -> Dict:
        """
        Run multiple Monte Carlo simulations.
        Returns aggregated results for analysis.
        """
        results = []
        
        # Run simulations with progress tracking
        for i in range(num_runs):
            result = self.run_single_simulation(hours_played)
            results.append(result)
            
            # Update progress bar if provided (update every 50 runs for smoother progress)
            if progress_bar is not None and (i + 1) % max(1, num_runs // 200) == 0:
                progress_percentage = (i + 1) / num_runs
                progress_bar.progress(progress_percentage)
        
        # Aggregate results
        final_bankrolls = [r['final_bankroll'] for r in results]
        trajectories = [r['trajectory'] for r in results]
        max_bankrolls = [r['max_bankroll'] for r in results]
        min_bankrolls = [r['min_bankroll'] for r in results]
        
        # Calculate statistics
        profits = [fb - self.calculator.starting_bankroll for fb in final_bankrolls]
        
        # Calculate average trajectory across all simulations
        max_length = max(len(traj) for traj in trajectories) if trajectories else 0
        avg_trajectory = []
        if max_length > 0:
            for hour in range(max_length):
                hour_values = []
                for traj in trajectories:
                    if hour < len(traj):
                        hour_values.append(traj[hour])
                    else:
                        # Use final value if trajectory ended early
                        hour_values.append(traj[-1])
                avg_trajectory.append(np.mean(hour_values))
        
        # Find best and worst trajectories for analysis
        best_trajectory = trajectories[np.argmax(final_bankrolls)] if trajectories else []
        worst_trajectory = trajectories[np.argmin(final_bankrolls)] if trajectories else []
        
        # Calculate actual average edge experienced across all simulations
        total_actual_edges = [r.get('actual_avg_edge', 0) for r in results]
        mean_actual_edge = np.mean(total_actual_edges) if total_actual_edges else 0
        
        aggregated_results = {
            'final_bankrolls': np.array(final_bankrolls),
            'trajectories': trajectories,
            'avg_trajectory': avg_trajectory,
            'best_trajectory': best_trajectory,
            'worst_trajectory': worst_trajectory,
            'profits': np.array(profits),
            'max_bankrolls': np.array(max_bankrolls),
            'min_bankrolls': np.array(min_bankrolls),
            'statistics': {
                'mean_profit': np.mean(profits),
                'median_profit': np.median(profits),
                'std_profit': np.std(profits),
                'min_profit': np.min(profits),
                'max_profit': np.max(profits),
                'prob_profit': np.mean(np.array(profits) > 0),
                'prob_ruin': np.mean(np.array(final_bankrolls) <= 0),
                'prob_double': np.mean(np.array(final_bankrolls) >= 2 * self.calculator.starting_bankroll),
                'mean_actual_edge': mean_actual_edge
            }
        }
        
        return aggregated_results
    
    def run_parallel_simulation(self, hours_played: int, num_runs: int = 1000, 
                              max_workers: int = 4) -> Dict:
        """
        Run Monte Carlo simulations in parallel for better performance.
        """
        def worker_simulation(run_batch_size):
            batch_results = []
            for _ in range(run_batch_size):
                result = self.run_single_simulation(hours_played)
                batch_results.append(result)
            return batch_results
        
        # Divide work into batches
        batch_size = max(1, num_runs // max_workers)
        batches = [batch_size] * (num_runs // batch_size)
        if num_runs % batch_size:
            batches.append(num_runs % batch_size)
        
        results = []
        
        # Run in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(worker_simulation, batch): batch 
                for batch in batches
            }
            
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_results = future.result()
                results.extend(batch_results)
        
        # Process results same as sequential version
        final_bankrolls = [r['final_bankroll'] for r in results]
        trajectories = [r['trajectory'] for r in results]
        max_bankrolls = [r['max_bankroll'] for r in results]
        min_bankrolls = [r['min_bankroll'] for r in results]
        
        profits = [fb - self.calculator.starting_bankroll for fb in final_bankrolls]
        
        aggregated_results = {
            'final_bankrolls': np.array(final_bankrolls),
            'trajectories': trajectories,
            'profits': np.array(profits),
            'max_bankrolls': np.array(max_bankrolls),
            'min_bankrolls': np.array(min_bankrolls),
            'statistics': {
                'mean_profit': np.mean(profits),
                'median_profit': np.median(profits),
                'std_profit': np.std(profits),
                'min_profit': np.min(profits),
                'max_profit': np.max(profits),
                'prob_profit': np.mean(np.array(profits) > 0),
                'prob_ruin': np.mean(np.array(final_bankrolls) <= 0),
                'prob_double': np.mean(np.array(final_bankrolls) >= 2 * self.calculator.starting_bankroll)
            }
        }
        
        return aggregated_results
    
    def analyze_variance_by_hours(self, max_hours: int, num_runs: int = 500) -> Dict:
        """
        Analyze how variance changes with different playing durations.
        """
        hour_intervals = [10, 25, 50, 100, 200, 500, max_hours]
        hour_intervals = [h for h in hour_intervals if h <= max_hours]
        
        variance_analysis = {}
        
        for hours in hour_intervals:
            results = self.run_simulation(hours, num_runs)
            variance_analysis[hours] = {
                'std_profit': np.std(results['profits']),
                'prob_ruin': results['statistics']['prob_ruin'],
                'mean_profit': results['statistics']['mean_profit']
            }
        
        return variance_analysis
