import numpy as np
import math

class BlackjackCalculator:
    """
    Core calculator for blackjack card counting simulation.
    Handles expected value, variance, and risk calculations.
    """
    
    def __init__(self, num_decks, starting_bankroll, hands_per_hour, betting_strategy):
        self.num_decks = num_decks
        self.starting_bankroll = starting_bankroll
        self.hands_per_hour = hands_per_hour
        self.betting_strategy = betting_strategy
        
        # Standard blackjack statistics
        self.std_dev_per_hand = 1.15  # Standard deviation per unit bet
        
        # True count frequency distribution (based on simulation studies)
        # Extended range from -3 to +6 to match betting strategy
        self.count_frequencies = {
            -3: 0.15, -2: 0.18, -1: 0.22, 0: 0.20, 1: 0.12,
            2: 0.08, 3: 0.03, 4: 0.015, 5: 0.005, 6: 0.0025
        }
        
        # Calculate edge based on true count (approximate)
        self.count_edges = {}
        for tc in range(-3, 7):  # TC from -3 to +6
            if tc <= 0:
                self.count_edges[tc] = -0.005 + (tc * 0.001)  # Slightly worse for negative counts
            else:
                self.count_edges[tc] = tc * 0.005  # ~0.5% per true count
        
        # Calculate weighted average edge and bet
        self.edge = self._calculate_weighted_edge()
        self.avg_bet = self._calculate_average_bet()
    
    def _get_bet_for_count(self, true_count):
        """
        Get the bet amount for a given true count based on betting strategy.
        """
        # Sort betting strategy by true count (descending) to find the right bet
        sorted_strategy = sorted(self.betting_strategy, key=lambda x: x['true_count'], reverse=True)
        
        for bet_rule in sorted_strategy:
            if true_count >= bet_rule['true_count']:
                return bet_rule['bet_amount']
        
        # If no rule matches, use the lowest bet
        return min(rule['bet_amount'] for rule in self.betting_strategy)
    
    def _calculate_weighted_edge(self):
        """
        Calculate weighted average edge based on true count frequencies and betting strategy.
        """
        total_weighted_edge = 0
        total_frequency = 0
        
        for true_count, frequency in self.count_frequencies.items():
            edge = self.count_edges[true_count]
            bet_amount = self._get_bet_for_count(true_count)
            
            # Weight the edge by frequency and bet size
            weighted_edge = frequency * edge * bet_amount
            total_weighted_edge += weighted_edge
            total_frequency += frequency * bet_amount
        
        return total_weighted_edge / total_frequency if total_frequency > 0 else 0
    
    def _calculate_average_bet(self):
        """
        Calculate average bet size based on true count frequencies and betting strategy.
        """
        total_weighted_bet = 0
        
        for true_count, frequency in self.count_frequencies.items():
            bet_amount = self._get_bet_for_count(true_count)
            total_weighted_bet += frequency * bet_amount
        
        return total_weighted_bet
    
    def calculate_hourly_ev(self):
        """Calculate expected value per hour."""
        hands_per_hour = self.hands_per_hour
        ev_per_hand = self.edge * self.avg_bet
        return ev_per_hand * hands_per_hour
    
    def calculate_hourly_std(self):
        """Calculate standard deviation per hour."""
        hands_per_hour = self.hands_per_hour
        variance_per_hand = (self.std_dev_per_hand * self.avg_bet) ** 2
        hourly_variance = variance_per_hand * hands_per_hour
        return math.sqrt(hourly_variance)
    
    def calculate_risk_of_ruin(self, hours_played):
        """
        Calculate risk of ruin using the standard formula for advantage play.
        RoR = exp(-2 * edge * bankroll / variance)
        """
        # Risk of ruin formula for dollar-based betting
        if self.edge <= 0:
            return 100.0  # Certain ruin with no edge
        
        # Calculate the effective edge per dollar bet
        # This accounts for the weighted average of all bet sizes
        effective_edge_per_dollar = self.edge
        
        # Standard deviation per dollar (using the average bet as normalization)
        std_dev_per_dollar = self.std_dev_per_hand / self.avg_bet if self.avg_bet > 0 else self.std_dev_per_hand
        
        # Calculate RoR using bankroll in dollars
        exponent = -2 * effective_edge_per_dollar * self.starting_bankroll / (std_dev_per_dollar ** 2 * self.avg_bet ** 2)
        ror = math.exp(exponent) * 100
        
        return min(ror, 100.0)  # Cap at 100%
    
    def calculate_recommended_bankroll(self):
        """
        Calculate recommended bankroll for given risk tolerance.
        Uses Kelly Criterion and risk of ruin formulas.
        """
        # Target risk of ruin of 5%
        target_ror = 0.05
        
        if self.edge <= 0:
            return float('inf')
        
        # Solve for bankroll: RoR = exp(-2 * edge * bankroll / variance)
        # bankroll = -ln(RoR) * variance / (2 * edge)
        variance_per_unit = self.std_dev_per_hand ** 2
        
        required_units = -math.log(target_ror) * variance_per_unit / (2 * self.edge)
        required_bankroll = required_units * self.avg_bet
        
        return required_bankroll
    
    def simulate_single_hand(self):
        """
        Simulate a single hand outcome using the betting strategy.
        Returns profit/loss for the hand.
        """
        # Use the more accurate count-based simulation
        return self.simulate_hand_with_count()
    
    def simulate_hand_with_count(self):
        """
        Simulate a single hand with realistic true count distribution.
        Returns profit/loss for the hand based on count frequencies.
        """
        # Randomly select true count based on frequencies
        rand = np.random.random()
        cumulative_freq = 0
        
        selected_tc = None
        for true_count in sorted(self.count_frequencies.keys()):
            cumulative_freq += self.count_frequencies[true_count]
            if rand <= cumulative_freq:
                selected_tc = true_count
                break
        
        if selected_tc is None:
            selected_tc = 0  # Default to neutral count
        
        # Get edge and bet for this count
        edge = self.count_edges[selected_tc]
        bet_amount = self._get_bet_for_count(selected_tc)
        
        # Simulate hand outcome with this edge
        outcome = np.random.normal(
            loc=edge * bet_amount,
            scale=self.std_dev_per_hand * bet_amount
        )
        
        return outcome
    
    def get_simulation_parameters(self):
        """Return key parameters for display."""
        min_bet = min(rule['bet_amount'] for rule in self.betting_strategy)
        max_bet = max(rule['bet_amount'] for rule in self.betting_strategy)
        
        return {
            'num_decks': self.num_decks,
            'bet_spread': f"${min_bet}-${max_bet}",
            'starting_bankroll': self.starting_bankroll,
            'calculated_edge': f"{self.edge*100:.3f}%",
            'hands_per_hour': self.hands_per_hour,
            'edge': self.edge,
            'avg_bet': self.avg_bet,
            'std_dev_per_hand': self.std_dev_per_hand,
            'betting_strategy': self.betting_strategy
        }
