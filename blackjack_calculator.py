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
        Uses professional estimates for rule impact on EV.
        """
        # Start with baseline 6-deck H17 house edge
        base_edge = 0.006  # ~0.6% baseline house edge
        
        # Apply rule adjustments based on professional estimates
        # Dealer rules
        if not self.game_rules['dealer_hits_soft_17']:  # S17
            base_edge -= 0.002  # Dealer stands on soft 17: +0.2% for player
        
        # Player-favorable rules
        if self.game_rules['double_after_split']:
            base_edge -= 0.0013  # DAS: +0.13% for player
        
        if self.game_rules['late_surrender']:
            base_edge -= 0.0009  # Late surrender: +0.08-0.1% for player
        
        if self.game_rules['can_split_aces']:
            if self.game_rules['resplit_aces']:
                base_edge -= 0.0008  # RSA: +0.07-0.1% for player
            # If no RSA but can split aces once, assume standard rules (no additional adjustment)
        else:
            base_edge += 0.0018  # No splitting aces: -0.18% for player
        
        # Multiple splits beyond 2 hands
        if self.game_rules['max_splits'] > 2:
            extra_splits = self.game_rules['max_splits'] - 2
            base_edge -= (extra_splits * 0.00015)  # +0.01-0.02% per extra hand
        
        # Deck count adjustments (industry standard)
        if self.num_decks == 1:
            base_edge -= 0.0048  # Single deck advantage
        elif self.num_decks == 2:
            base_edge -= 0.0025  # Double deck advantage
        elif self.num_decks == 4:
            base_edge -= 0.0006  # Four deck slight advantage
        elif self.num_decks == 8:
            base_edge += 0.0002  # Eight deck slight disadvantage
        
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
            # Maximum penetration (all cards played)
            filename = f"true count distributions/{self.num_decks}decks-nopenetration.csv"
        else:
            # Specific penetration (only penetration_deck out of num_decks played)
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
                
                # Map all counts to betting strategy ranges
                # Extreme counts get mapped to the edge cases of our strategy
                mapped_frequencies = {}
                
                # Initialize our strategy range with zeros
                for tc in range(-3, 7):
                    mapped_frequencies[tc] = 0.0
                
                # Map all frequencies from CSV to our strategy range
                for true_count, freq in frequencies.items():
                    # freq is already converted from percentage in the CSV loading
                    
                    if true_count <= -3:
                        # All very negative counts map to TC -3
                        mapped_frequencies[-3] += freq
                    elif true_count >= 6:
                        # All very positive counts map to TC +6  
                        mapped_frequencies[6] += freq
                    elif -3 < true_count < 6:
                        # Counts in our range map directly
                        mapped_frequencies[true_count] = freq
                
                return mapped_frequencies
                    
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
        Applies table rules adjustments to professional edge values.
        For extreme counts, we need to use realistic edges, not just clamp to TC +6.
        """
        # Base edges from professional simulation data (6-deck, H17 baseline)
        base_edges = {
            -3: -0.006,  # -0.6% (includes all counts <= -3)
            -2: -0.005,  # -0.5%
            -1: -0.004,  # -0.4%
             0: -0.003,  # -0.3%
             1:  0.005,  # +0.5%
             2:  0.010,  # +1.0%
             3:  0.015,  # +1.5%
             4:  0.020,  # +2.0%
             5:  0.022,  # +2.2%
             6:  0.025,  # +2.5% (represents average edge for all counts >= +6)
        }
        
        # Calculate rule adjustments (difference from baseline)
        rule_adjustment = self.base_house_edge - 0.006  # Difference from 6-deck H17 baseline
        
        # Apply rule adjustments to all true counts
        edges = {}
        for tc, base_edge in base_edges.items():
            # Apply the rule adjustment to shift all edges
            adjusted_edge = base_edge - rule_adjustment
            
            # Additional adjustments for specific rules at positive counts
            if tc > 0:
                # Late surrender is most valuable at high counts
                if self.game_rules['late_surrender']:
                    surrender_bonus = min(tc * 0.0002, 0.0008)  # Scales with TC, max 0.08%
                    adjusted_edge += surrender_bonus
                
                # DAS provides slight advantage across positive counts
                if self.game_rules['double_after_split']:
                    das_bonus = min(tc * 0.0001, 0.0003)  # Scales with TC, max 0.03%
                    adjusted_edge += das_bonus
                
                # RSA provides small advantage at positive counts
                if self.game_rules['can_split_aces'] and self.game_rules['resplit_aces']:
                    rsa_bonus = min(tc * 0.00005, 0.0002)  # Scales with TC, max 0.02%
                    adjusted_edge += rsa_bonus
            
            edges[tc] = adjusted_edge
        
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
        total_weighted_frequency = 0
        
        for true_count, frequency in self.count_frequencies.items():
            edge = self.count_edges[true_count]
            bet_amount = self._get_bet_for_count(true_count)
            
            # Only include hands where we actually bet (bet_amount > 0)
            if bet_amount > 0:
                # Weight the edge by frequency and bet size for proper averaging
                weighted_edge = frequency * edge * bet_amount
                total_weighted_edge += weighted_edge
                total_weighted_frequency += frequency * bet_amount
        
        return total_weighted_edge / total_weighted_frequency if total_weighted_frequency > 0 else 0
    
    def _calculate_average_bet(self):
        """
        Calculate average bet size based on true count frequencies and betting strategy.
        Uses consistent weighting with EV calculation.
        """
        total_bet = 0
        total_freq = 0
        
        for true_count, frequency in self.count_frequencies.items():
            bet_amount = self._get_bet_for_count(true_count)
            total_bet += bet_amount * frequency
            total_freq += frequency
        
        return total_bet / total_freq if total_freq > 0 else 0
    
    def _get_actual_edge_for_count(self, true_count):
        """
        Get the actual edge for any true count, including extreme values.
        Don't clamp edge values - use realistic edges for extreme counts.
        """
        if true_count <= -3:
            return self.count_edges[-3]  # Use TC-3 edge for very negative counts
        elif true_count >= 6:
            # For very high counts, use escalating edge (roughly +0.5% per TC above +6)
            extra_counts = true_count - 6
            base_edge = self.count_edges[6]  # +2.5% for TC+6
            return base_edge + (extra_counts * 0.005)  # +0.5% per additional TC
        else:
            return self.count_edges[true_count]
    
    def _calculate_ev_per_hand(self):
        """
        Calculate EV per hand using actual CSV frequencies and proper edge values.
        This properly handles extreme counts without artificial clamping.
        """
        total_ev = 0
        total_freq = 0
        
        # Use the same mapped frequencies that the simulation uses
        for true_count, frequency in self.count_frequencies.items():
            # Get edge and bet for this count
            edge = self.count_edges[true_count]
            bet_amount = self._get_bet_for_count(true_count)
            
            # Include ALL hands (even when we sit out with $0 bet)
            ev_contribution = edge * bet_amount * frequency
            total_ev += ev_contribution
            total_freq += frequency
        
        ev_per_hand = total_ev / total_freq if total_freq > 0 else 0
        
        return ev_per_hand
    
    def calculate_hourly_ev(self):
        """Calculate expected value per hour."""
        ev_per_hand = self._calculate_ev_per_hand()
        hands_per_hour = self.hands_per_hour
        return ev_per_hand * hands_per_hour
    
    def calculate_hourly_std(self):
        """Calculate standard deviation per hour using bet-weighted variance."""
        hourly_variance = self.calculate_hourly_variance()
        return math.sqrt(hourly_variance)
    
    def calculate_risk_of_ruin(self, hours_played):
        """
        Calculate risk of ruin using proper bet-weighted edge and realistic variance.
        Returns theoretical RoR that should approximate Monte Carlo results.
        """
        # Calculate true bet-weighted edge (total EV / total money wagered)
        total_weighted_edge = 0
        total_bet_amount = 0
        
        for true_count, frequency in self.count_frequencies.items():
            edge = self.count_edges[true_count]
            bet_amount = self._get_bet_for_count(true_count)
            
            if bet_amount > 0:  # Only count hands where we actually bet
                total_weighted_edge += edge * bet_amount * frequency
                total_bet_amount += bet_amount * frequency
        
        if total_bet_amount == 0:
            return 100.0
        
        # True bet-weighted edge
        weighted_edge = total_weighted_edge / total_bet_amount
        
        # If no positive edge, immediate ruin
        if weighted_edge <= 0:
            return 100.0
        
        # Calculate average bet per hand (including hands where we sit out)
        # total_bet_amount is sum of (bet * frequency), we need to divide by total frequency
        total_frequency = sum(self.count_frequencies.values())
        average_bet = total_bet_amount / total_frequency if total_frequency > 0 else 0
        
        if average_bet <= 0:
            return 100.0
        
        # Number of betting units in bankroll
        betting_units = self.starting_bankroll / average_bet
        
        # Use realistic variance for modern blackjack with doubling/splitting
        # Higher variance for strategies with bet spreads due to more aggressive play at high counts
        empirical_variance = 1.6  # More realistic than 1.3 for modern games with full basic strategy
        
        # Risk of ruin formula: RoR = exp(-2 * edge * units / variance)
        exponent = -2 * weighted_edge * betting_units / empirical_variance
        
        # Calculate natural risk without artificial bounds
        ror = math.exp(exponent) * 100
        
        return round(min(ror, 100.0), 1)
    
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
