"""
Data preprocessing logic for ATP tennis matches.
Handles missing values, player order randomization, and target generation.
"""
import config
import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame, odds_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Preprocess raw ATP match data by:
    - Handling missing values
    - Randomizing player order to create a balanced dataset
    - Creating a binary target (1 = P1 win, 0 = P1 loss)
    - Parsing score strings into game/set statistics
    """
    df = df.copy()

    # Handle missing values
    df["winner_rank"] = df["winner_rank"].fillna(config.DEFAULT_RANK)
    df["loser_rank"]  = df["loser_rank"].fillna(config.DEFAULT_RANK)

    df["winner_age"] = df["winner_age"].fillna(df["winner_age"].median())
    df["loser_age"]  = df["loser_age"].fillna(df["loser_age"].median())

    # ----------------------------------------------------
    # Merge Odds Data
    # ----------------------------------------------------
    if odds_df is not None and not odds_df.empty:
        odds_df = odds_df.copy()
        
        # Sackmann date is YYYYMMDD
        df_dates = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
        df["_year"] = df_dates.dt.year
        df["_month"] = df_dates.dt.month
        
        # Last name extraction: "Grigor Dimitrov" -> "Dimitrov"
        df["_w_last"] = df["winner_name"].str.split().str[-1]
        df["_l_last"] = df["loser_name"].str.split().str[-1]

        # tennis-data date is e.g. M/D/YY or MM/DD/YYYY
        if "Date" in odds_df.columns:
            o_dates = pd.to_datetime(odds_df["Date"], errors="coerce")
            odds_df["_year"] = o_dates.dt.year
            odds_df["_month"] = o_dates.dt.month
        else:
            odds_df["_year"] = np.nan
            odds_df["_month"] = np.nan
            
        # Last name extraction: "Dimitrov G." -> "Dimitrov"
        if "Winner" in odds_df.columns:
            odds_df["_w_last"] = odds_df["Winner"].str.split().str[0]
        else:
            odds_df["_w_last"] = np.nan
            
        if "Loser" in odds_df.columns:
            odds_df["_l_last"] = odds_df["Loser"].str.split().str[0]
        else:
            odds_df["_l_last"] = np.nan
            
        # Extract best odds (Avg > B365)
        if "AvgW" in odds_df.columns and "B365W" in odds_df.columns:
            odds_df["_odds_w"] = odds_df["AvgW"].fillna(odds_df["B365W"])
        elif "AvgW" in odds_df.columns:
            odds_df["_odds_w"] = odds_df["AvgW"]
        elif "B365W" in odds_df.columns:
            odds_df["_odds_w"] = odds_df["B365W"]
        else:
            odds_df["_odds_w"] = np.nan
            
        if "AvgL" in odds_df.columns and "B365L" in odds_df.columns:
            odds_df["_odds_l"] = odds_df["AvgL"].fillna(odds_df["B365L"])
        elif "AvgL" in odds_df.columns:
            odds_df["_odds_l"] = odds_df["AvgL"]
        elif "B365L" in odds_df.columns:
            odds_df["_odds_l"] = odds_df["B365L"]
        else:
            odds_df["_odds_l"] = np.nan
            
        # Deduplicate keys
        odds_keys = ["_year", "_month", "_w_last", "_l_last"]
        odds_to_merge = odds_df[odds_keys + ["_odds_w", "_odds_l"]].drop_duplicates(subset=odds_keys)
        
        df = df.merge(odds_to_merge, on=odds_keys, how="left")
        
        # Clean up temp keys
        df.drop(columns=["_year", "_month", "_w_last", "_l_last"], inplace=True)
    else:
        df["_odds_w"] = np.nan
        df["_odds_l"] = np.nan

    # Randomise player order
    rng = np.random.default_rng(seed=42)
    swap_players = rng.random(len(df)) > config.DEFAULT_WIN_PCT

    # Build Player 1 / Player 2 dataset
    new_df = pd.DataFrame({
        "tourney_date": df["tourney_date"],
        "surface": df["surface"],
        "tourney_level": df["tourney_level"],

        "p1_name": np.where(swap_players, df["loser_name"], df["winner_name"]),
        "p1_rank": np.where(swap_players, df["loser_rank"], df["winner_rank"]),
        "p1_age":  np.where(swap_players, df["loser_age"],  df["winner_age"]),
        "p1_odds": np.where(swap_players, df["_odds_l"], df["_odds_w"]),

        "p2_name": np.where(swap_players, df["winner_name"], df["loser_name"]),
        "p2_rank": np.where(swap_players, df["winner_rank"], df["loser_rank"]),
        "p2_age":  np.where(swap_players, df["winner_age"],  df["loser_age"]),
        "p2_odds": np.where(swap_players, df["_odds_w"],  df["_odds_l"]),
    })

    # Target: did Player 1 win?
    new_df["target"] = (~swap_players).astype(int)

    # ----------------------------------------------------
    # Parse scores to get games/sets won/lost
    # ----------------------------------------------------
    # We need the score from perspective of the OFFICIAL winner first, 
    # then swap if p1 is actually the loser.
    
    # 1. Parse score for the official winner/loser
    scores = df["score"].fillna("")
    w_games_won, w_games_lost = [], []
    w_sets_won, w_sets_lost = [], []

    for score in scores:
        gw, gl, sw, sl = parse_match_score(score)
        w_games_won.append(gw)
        w_games_lost.append(gl)
        w_sets_won.append(sw)
        w_sets_lost.append(sl)

    # 2. Assign to p1 / p2 based on swap_players
    # If swap_players is True: P1 is Loser, P2 is Winner
    # If swap_players is False: P1 is Winner, P2 is Loser
    
    p1_games_won = np.where(swap_players, w_games_lost, w_games_won)
    p1_games_lost = np.where(swap_players, w_games_won, w_games_lost)
    p1_sets_won  = np.where(swap_players, w_sets_lost, w_sets_won)
    p1_sets_lost = np.where(swap_players, w_sets_won, w_sets_lost)

    p2_games_won = np.where(swap_players, w_games_won, w_games_lost)
    p2_games_lost = np.where(swap_players, w_games_lost, w_games_won)
    p2_sets_won  = np.where(swap_players, w_sets_won, w_sets_lost)
    p2_sets_lost = np.where(swap_players, w_sets_lost, w_sets_won)
    
    new_df["p1_games_won"] = p1_games_won
    new_df["p1_games_lost"] = p1_games_lost
    new_df["p1_sets_won"] = p1_sets_won
    new_df["p1_sets_lost"] = p1_sets_lost
    
    new_df["p2_games_won"] = p2_games_won
    new_df["p2_games_lost"] = p2_games_lost
    new_df["p2_sets_won"] = p2_sets_won
    new_df["p2_sets_lost"] = p2_sets_lost

    return new_df

def parse_match_score(score_str: str) -> tuple[int, int, int, int]:
    """
    Parses a score string (e.g., "6-4 3-6 7-6(5)") 
    Returns: (winner_games, loser_games, winner_sets, loser_sets)
    """
    w_games = 0
    l_games = 0
    w_sets = 0
    l_sets = 0
    
    if not isinstance(score_str, str) or not score_str.strip():
        return 0, 0, 0, 0
        
    # Remove retirement/walkover markers if easy to handle, 
    # though usually they are just "6-4 2-0 RET"
    # We'll just split by space
    sets = score_str.split()
    
    for s in sets:
        # Handle "RET", "W/O" etc
        if "RET" in s or "W/O" in s or "def." in s:
            continue
            
        # Remove tiebreak scores like 7-6(4) -> 7-6
        if "(" in s:
            s = s.split("(")[0]
            
        parts = s.split("-")
        if len(parts) != 2:
            continue
            
        try:
            w_g = int(parts[0])
            l_g = int(parts[1])
            
            w_games += w_g
            l_games += l_g
            
            if w_g > l_g:
                w_sets += 1
            elif l_g > w_g:
                l_sets += 1
        except ValueError:
            pass
            
    return w_games, l_games, w_sets, l_sets