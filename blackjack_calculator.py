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
        
        # True count frequencies from CSV data
        self.count_frequencies = self._load_count_frequencies_from_csv()
        
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
    
    def _load_count_frequencies_from_csv(self):
        """
        Load true count frequencies from CSV simulation data.
        """
        import os
        import csv
        
        # Determine the correct CSV file based on deck count and penetration
        penetration_deck = self.table_rules['penetration_deck']
        
        if penetration_deck == self.num_decks:
            # No penetration case
            filename = f"true count distributions/{self.num_decks}decks-nopenetration.csv"
        else:
            # Specific penetration
            filename = f"true count distributions/{self.num_decks}decks-{penetration_deck}penetration.csv"
        
        frequencies = {}
        
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        # Skip comment lines and headers
                        if row and not row[0].startswith('#') and row[0] != 'True Count':
                            try:
                                true_count = int(row[0])
                                percentage = float(row[1])
                                # Convert percentage to frequency (divide by 100)
                                frequencies[true_count] = percentage / 100.0
                            except (ValueError, IndexError):
                                continue
                
                # Filter to our range of interest (-3 to +6)
                filtered_frequencies = {}
                for tc in range(-3, 7):
                    if tc in frequencies:
                        filtered_frequencies[tc] = frequencies[tc]
                    else:
                        # Use a small default value for missing counts
                        filtered_frequencies[tc] = 0.001
                
                # Normalize to ensure they sum to 1.0
                total = sum(filtered_frequencies.values())
                if total > 0:
                    return {tc: freq/total for tc, freq in filtered_frequencies.items()}
                    
            except Exception as e:
                print(f"Error loading CSV {filename}: {e}")
        
        # Fallback to default frequencies if CSV loading fails
        return {
            -3: 0.12, -2: 0.15, -1: 0.20, 0: 0.24, 1: 0.16,
            2: 0.09, 3: 0.03, 4: 0.01, 5: 0.003, 6: 0.001
        }
    
    def _calculate_count_edges(self):
        """
        Calculate player edge for each true count based on simulation data.
        Uses professional simulation values for 6-deck, H17, 83% penetration.
        """
        # Base edges from professional simulation data
        base_edges = {
            -3: -0.006,  # -0.6%
            -2: -0.005,  # -0.5%
            -1: -0.004,  # -0.4%
             0: -0.003,  # -0.3%
             1:  0.005,  # +0.5%
             2:  0.010,  # +1.0%
             3:  0.015,  # +1.5%
             4:  0.020,  # +2.0%
             5:  0.022,  # +2.2%
             6:  0.023,  # +2.3%
        }
        
        # Adjust for table rules
        edges = {}
        for tc, base_edge in base_edges.items():
            edge = base_edge
            
            # Adjust for favorable rules (only for positive counts)
            if tc > 0:
                if self.game_rules['late_surrender']:
                    edge += 0.001  # Surrender adds ~0.1% at high counts
                
                if self.game_rules['double_after_split']:
                    edge += 0.0005  # DAS adds slight advantage
            
            edges[tc] = edge
        
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
        """Calculate standard deviation per hour using bet-weighted variance."""
        hourly_variance = self.calculate_hourly_variance()
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
