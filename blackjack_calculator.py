import numpy as np
import math

class BlackjackCalculator:
    """
    Core calculator for blackjack card counting simulation.
    Handles expected value, variance, and risk calculations.
    """
    
    def __init__(self, num_decks, starting_bankroll, hands_per_hour, betting_strategy, table_rules=None):
        self.num_decks = num_decks
        self.starting_bankroll = starting_bankroll
        self.hands_per_hour = hands_per_hour
        self.betting_strategy = betting_strategy
        
        # Default table rules if none provided
        if table_rules is None:
            table_rules = {
                'penetration_deck': 5,
                'dealer_hits_soft_17': True,
                'double_after_split': True,
                'can_split_aces': True,
                'resplit_aces': False,
                'max_splits': 3,
                'surrender_allowed': False
            }
        
        self.table_rules = table_rules
        
        # Configure game rules
        self.game_rules = {
            'decks': num_decks,
            'penetration': table_rules['penetration_deck'] / num_decks,
            'dealer_hits_soft_17': table_rules['dealer_hits_soft_17'],
            'blackjack_payout': 1.5,     # 3:2 payout
            'double_any_two': True,      # Can double on any two cards
            'double_after_split': table_rules['double_after_split'],
            'can_split_aces': table_rules['can_split_aces'],
            'resplit_aces': table_rules['resplit_aces'],
            'max_splits': table_rules['max_splits'],
            'late_surrender': table_rules['surrender_allowed'],
            'european_no_hole_card': False,  # Assume US rules
            'insurance_offered': True
        }
        
        # Calculate base house edge based on rules
        self.base_house_edge = self._calculate_base_house_edge()
        
        # Standard blackjack statistics  
        self.std_dev_per_hand = 0.95  # Standard deviation per unit bet (realistic for skilled play)
        
        # True count frequencies based on table rules and penetration
        self.count_frequencies = self._get_count_frequencies_for_rules()
        
        # Calculate edge for each true count
        self.count_edges = self._calculate_count_edges()
        
        # Calculate weighted average edge and bet
        self.edge = self._calculate_weighted_edge()
        self.avg_bet = self._calculate_average_bet()
    
    def _calculate_base_house_edge(self):
        """
        Calculate base house edge based on table rules.
        Based on composition-dependent basic strategy.
        """
        base_edge = 0.005  # Start with basic 6-deck H17 edge
        
        # Adjust for dealer rules
        if self.game_rules['dealer_hits_soft_17']:
            base_edge += 0.0022  # H17 increases house edge
        
        # Adjust for player-favorable rules
        if self.game_rules['double_after_split']:
            base_edge -= 0.0014  # DAS reduces house edge
        
        if self.game_rules['late_surrender']:
            base_edge -= 0.0008  # Late surrender reduces house edge
        
        if self.game_rules['can_split_aces']:
            base_edge -= 0.0005  # Split aces reduces house edge
            
        if self.game_rules['resplit_aces']:
            base_edge -= 0.0003  # Resplit aces further reduces edge
        
        # Adjust for number of decks (more decks = worse for player)
        if self.num_decks == 1:
            base_edge -= 0.0048
        elif self.num_decks == 2:
            base_edge -= 0.0025
        elif self.num_decks == 4:
            base_edge -= 0.0006
        elif self.num_decks == 8:
            base_edge += 0.0002
        
        return max(base_edge, 0.001)  # Minimum 0.1% house edge
    
    def _get_count_frequencies_for_rules(self):
        """
        Get true count frequencies based on table rules and penetration.
        Based on simulation studies from Schlesinger and Wattenberger.
        """
        penetration = self.game_rules['penetration']
        
        # Base frequencies for 75% penetration, 6-deck game
        if penetration >= 0.80:  # Deep penetration (80%+)
            frequencies = {
                -3: 0.08, -2: 0.12, -1: 0.18, 0: 0.22, 1: 0.20,
                2: 0.12, 3: 0.06, 4: 0.02, 5: 0.008, 6: 0.002
            }
        elif penetration >= 0.70:  # Good penetration (70-80%)
            frequencies = {
                -3: 0.12, -2: 0.15, -1: 0.20, 0: 0.24, 1: 0.16,
                2: 0.09, 3: 0.03, 4: 0.01, 5: 0.003, 6: 0.001
            }
        else:  # Poor penetration (<70%)
            frequencies = {
                -3: 0.15, -2: 0.18, -1: 0.22, 0: 0.26, 1: 0.12,
                2: 0.06, 3: 0.02, 4: 0.005, 5: 0.002, 6: 0.001
            }
        
        # Adjust for number of decks
        if self.num_decks <= 2:
            # Single/double deck: more extreme counts
            for tc in frequencies:
                if abs(tc) >= 2:
                    frequencies[tc] *= 1.3
                else:
                    frequencies[tc] *= 0.9
        elif self.num_decks >= 8:
            # 8-deck: fewer extreme counts
            for tc in frequencies:
                if abs(tc) >= 2:
                    frequencies[tc] *= 0.7
                else:
                    frequencies[tc] *= 1.1
        
        # Normalize to ensure they sum to 1.0
        total = sum(frequencies.values())
        return {tc: freq/total for tc, freq in frequencies.items()}
    
    def _calculate_count_edges(self):
        """
        Calculate player edge for each true count based on table rules.
        """
        edges = {}
        
        # Base edge per true count (affected by rules)
        edge_per_count = 0.005  # Standard 0.5% per true count
        
        # Adjust edge per count based on rules
        if self.game_rules['late_surrender']:
            edge_per_count += 0.0005  # Surrender increases advantage at high counts
        
        if self.game_rules['double_after_split']:
            edge_per_count += 0.0002  # DAS slightly increases advantage
        
        for tc in range(-3, 7):
            if tc <= 0:
                # Negative/neutral counts: house edge gets worse for player
                edges[tc] = -self.base_house_edge + (tc * edge_per_count)
            else:
                # Positive counts: player advantage increases
                edges[tc] = -self.base_house_edge + (tc * 0.007)  # Slightly higher for positive
        
        return edges
    
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
        Calculate risk of ruin using the correct formula for dollar-based betting.
        Uses Kelly Criterion approach for accurate risk assessment.
        """
        if self.edge <= 0:
            return 100.0  # Certain ruin with no edge
        
        # Calculate average bet size (in dollars)
        avg_bet_dollars = self._calculate_average_bet()
        
        # Convert edge to edge per dollar bet
        edge_per_dollar = self.edge / avg_bet_dollars if avg_bet_dollars > 0 else 0
        
        # Calculate variance per dollar bet (standard deviation squared)
        variance_per_dollar = self.std_dev_per_hand ** 2
        
        # Number of betting units in bankroll
        betting_units = self.starting_bankroll / avg_bet_dollars if avg_bet_dollars > 0 else 0
        
        # Risk of ruin formula for advantage play: exp(-2 * edge_per_unit * units / variance_per_unit)
        if betting_units > 0 and variance_per_dollar > 0:
            exponent = -2 * edge_per_dollar * betting_units / variance_per_dollar
            ror = math.exp(exponent) * 100
        else:
            ror = 0.0
        
        return min(ror, 100.0)  # Cap at 100%
    
    def calculate_hourly_variance(self):
        """Calculate variance per hour based on betting strategy."""
        total_variance = 0
        for tc in range(-3, 7):
            frequency = self.count_frequencies.get(tc, 0)
            bet_amount = self._get_bet_for_count(tc)
            
            if bet_amount > 0:  # Only count hands where we actually bet
                hand_variance = (self.std_dev_per_hand * bet_amount) ** 2
                total_variance += frequency * hand_variance * self.hands_per_hour
        
        return total_variance
    
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
