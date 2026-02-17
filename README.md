# Tennis EV Trading System 🎾💰

A machine learning pipeline for evaluating ATP singles matches and identifying positive expected value (EV) betting opportunities using historical match data and bookmaker odds.

This project extends traditional outcome prediction by integrating historical odds, EV calculations, and bankroll simulation to assess whether an edge exists against the market.

It includes an interactive CLI that computes the predicted win probability for a matchup, compares it to bookmaker odds, and outputs expected value along with recommended bet sizing using Kelly criterion simulation.

- **Data Source:** Jeff Sackmann's [ATP Matches Dataset](https://github.com/JeffSackmann/tennis_atp) and historical bookmaker odds
- **Tech Stack:** Python, Pandas, Scikit-Learn, XGBoost, Matplotlib
- **Best Model:** Random Forest (~64% Accuracy; ROI-focused evaluation on 2014–2024 test set)

## Features Engineered

All features are computed chronologically, using only information available prior to each match:

1. **Surface-Specific Win Rate:** Player performance on Clay/Hard/Grass.
2. **Head-to-Head (H2H):** Historical dominance between two players.
3. **Dynamic Rankings:** Player rank at the time of the match.
4. **Historical Odds:** Bookmaker odds history for each matchup.
5. **Expected Value (EV):** Comparison of model-predicted win probability vs. bookmaker odds.
6. **ROI Metrics & Bankroll Simulation:** Analyze profitability of betting strategies and simulate Kelly criterion-based bankroll growth.
7. **Calibration Metrics:** Assess how well predicted probabilities reflect actual outcomes.

## How to Run

1. Install dependencies:
   ```bash
   pip install pandas scikit-learn xgboost matplotlib seaborn
   ```
2. Set up your virtual environment: source venv/bin/activate
3. Run the script: python tennis_ev.py
4. The script will:

- Download ATP match data (cached locally) and historical odds

- Engineer features

- Train predictive models (Random Forest, XGBoost, etc.)

- Evaluate ROI, calibration, and expected value metrics

- Simulate bankroll growth using Kelly criterion

- Launch an interactive CLI to assess EV for custom matchups
