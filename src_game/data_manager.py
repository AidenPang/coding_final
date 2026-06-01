import os
import pandas as pd
from datetime import datetime

class DataManager:
    def __init__(self, filepath='data/leaderboard.csv'):
        self.filepath = filepath
        self.headers = ['Timestamp', 'PlayerName', 'Score', 'Streak', 'GameMode', 'TimeElapsed']
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Ensures that the directory and the CSV file exist with correct headers."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            self._create_empty_csv()
            
    def _create_empty_csv(self):
        """Creates an empty CSV file with the headers."""
        try:
            df = pd.DataFrame(columns=self.headers)
            df.to_csv(self.filepath, index=False)
        except Exception as e:
            print(f"Error creating leaderboard file: {e}")

    def add_score(self, player_name, score, streak, game_mode, time_elapsed):
        """
        Adds a new record to the leaderboard CSV.
        Handles FileI/O and exceptions.
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Simple sanitization
            player_name = str(player_name).strip()
            if not player_name:
                player_name = "Anonymous"
                
            new_record = {
                'Timestamp': timestamp,
                'PlayerName': player_name,
                'Score': int(score),
                'Streak': int(streak),
                'GameMode': str(game_mode),
                'TimeElapsed': float(time_elapsed)
            }
            
            # Load existing records, append new one, and write back
            # Handles case where file exists but might be corrupted
            try:
                df = pd.read_csv(self.filepath)
            except Exception:
                # If corrupted, reset with headers
                df = pd.DataFrame(columns=self.headers)
                
            new_row_df = pd.DataFrame([new_record])
            df = pd.concat([df, new_row_df], ignore_index=True)
            df.to_csv(self.filepath, index=False)
            return True
        except Exception as e:
            print(f"Error saving score: {e}")
            return False

    def get_leaderboard(self, game_mode=None, limit=10):
        """
        Loads the leaderboard using Pandas, sorts it by Score (descending) and Streak (descending),
        and returns the top records.
        """
        try:
            df = pd.read_csv(self.filepath)
            if df.empty:
                return pd.DataFrame(columns=self.headers)
                
            if game_mode:
                df = df[df['GameMode'] == game_mode]
                
            # Sort by Score desc, then Streak desc, then TimeElapsed asc
            df_sorted = df.sort_values(
                by=['Score', 'Streak', 'TimeElapsed'], 
                ascending=[False, False, True]
            )
            
            # Add a rank column
            df_sorted.insert(0, 'Rank', range(1, len(df_sorted) + 1))
            
            return df_sorted.head(limit)
        except Exception as e:
            print(f"Error reading leaderboard: {e}")
            return pd.DataFrame(columns=self.headers)

    def get_analytics(self):
        """
        Calculates aggregate statistics using Pandas.
        Returns:
            dict containing counts, averages, and maxes.
        """
        stats = {
            'total_games': 0,
            'avg_score': 0.0,
            'max_score': 0,
            'max_streak': 0,
            'avg_time_elapsed': 0.0,
            'popular_mode': 'None'
        }
        try:
            df = pd.read_csv(self.filepath)
            if df.empty:
                return stats
                
            stats['total_games'] = int(df.shape[0])
            stats['avg_score'] = float(df['Score'].mean())
            stats['max_score'] = int(df['Score'].max())
            stats['max_streak'] = int(df['Streak'].max())
            stats['avg_time_elapsed'] = float(df['TimeElapsed'].mean())
            
            # Find the most popular game mode
            if not df['GameMode'].empty:
                stats['popular_mode'] = str(df['GameMode'].value_counts().idxmax())
                
            return stats
        except Exception as e:
            print(f"Error compiling analytics: {e}")
            return stats
            
    def get_raw_csv_path(self):
        return self.filepath
