import numpy as np
import math

class BlackjackCalculator:
    """
    Core calculator for blackjack card counting simulation.
    Handles expected value, variance, and risk calculations.
    """
    
    def __init__(self, num_decks, min_bet, max_bet, starting_bankroll, 
                 win_rate_percent, hands_per_hour):
        self.num_decks = num_decks
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.starting_bankroll = starting_bankroll
        self.win_rate_percent = win_rate_percent
        self.hands_per_hour = hands_per_hour
        
        # Standard blackjack statistics
        self.base_house_edge = 0.005  # 0.5% house edge for basic strategy
        self.std_dev_per_hand = 1.15  # Standard deviation per unit bet
        
        # Calculate effective edge
        self.edge = (win_rate_percent / 100) - self.base_house_edge
        
        # Average bet calculation (simplified model)
        self.avg_bet = self._calculate_average_bet()
    
    def _calculate_average_bet(self):
        """
        Calculate average bet size based on spread.
        Assumes a simple linear progression based on count.
        """
        # Simplified model: assume even distribution across bet range
        # In reality, this would depend on the specific counting system
        return (self.min_bet + self.max_bet) / 2
    
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
        total_hands = hours_played * self.hands_per_hour
        
        # Calculate total variance
        variance_per_hand = (self.std_dev_per_hand * self.avg_bet) ** 2
        total_variance = variance_per_hand * total_hands
        
        # Risk of ruin formula
        if self.edge <= 0:
            return 100.0  # Certain ruin with no edge
        
        # Convert bankroll to units
        bankroll_units = self.starting_bankroll / self.avg_bet
        
        # Calculate RoR
        exponent = -2 * self.edge * bankroll_units / (self.std_dev_per_hand ** 2)
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
        Simulate a single hand outcome.
        Returns profit/loss for the hand.
        """
        # Determine bet size (simplified - in reality depends on count)
        bet_size = np.random.choice([self.min_bet, self.max_bet], 
                                  p=[0.7, 0.3])  # More small bets than large
        
        # Simulate hand outcome
        # Use normal distribution centered on expected value
        outcome = np.random.normal(
            loc=self.edge * bet_size,
            scale=self.std_dev_per_hand * bet_size
        )
        
        return outcome
    
    def get_simulation_parameters(self):
        """Return key parameters for display."""
        return {
            'num_decks': self.num_decks,
            'bet_spread': f"{self.min_bet}-{self.max_bet}",
            'starting_bankroll': self.starting_bankroll,
            'win_rate_percent': self.win_rate_percent,
            'hands_per_hour': self.hands_per_hour,
            'edge': self.edge,
            'avg_bet': self.avg_bet,
            'std_dev_per_hand': self.std_dev_per_hand
        }
