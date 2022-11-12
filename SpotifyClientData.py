import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import List, Dict, Union

################################################################################

class SpotifyClientData:
    """Class for using the Spotify API to retrieve store data on a Spotify
    user's listening habits.
    """
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self._client = spotipy.Spotify(
            auth_manager=SpotifyOAuth(client_id=client_id,
                                      client_secret=client_secret,
                                      redirect_uri=redirect_uri,
                                      scope=["user-library-read", "user-top-read"])
                        )
        
        self._client_name = self._client.current_user()['display_name']
        
        self._top_artists = [
            {'uri': artist['uri'], 'name': artist['name'],
            'popularity': artist['popularity'], 'genres': artist['genres']} \
                for artist in \
                self._client.current_user_top_artists(time_range='long_term',
                                                      limit=50)['items']
        ]

        self._top_tracks = [
            {'uri': track['uri'], 'name': track['name'], 'popularity': track['popularity'],
             'length_ms': track['duration_ms'],
             'audio_features': self._get_audio_features(track['uri'])} for track in \
                self._client.current_user_top_tracks(time_range='long_term',
                                                    limit=50)['items']
        ]

    def _get_audio_features(self, uri: str) -> Dict[str, float]:
        features = self._client.audio_features([uri])[0]

        return {'acousticness': features['acousticness'],
                'danceability': features['danceability'],
                'energy': features['energy'],
                'instrumentalness': features['instrumentalness'],
                # divide loudness by -60 so it's on a scale from 0.0 - 1.0
                'loudness': features['loudness'] / -60.0,
                'mode': features['mode'],
                'speechiness': features['speechiness'],
                'valence': features['valence']}

    def get_client_name(self) -> str:
        return self._client_name

    def get_top_artists(self) -> List[Dict[str, Union[str, int, List[str]]]]:
        return self._top_artists

    def get_top_tracks(self) -> List[Dict[str, Union[str, int, Dict[str, float]]]]:
        return self._top_tracks

# Get favourite genres of user, based on top longterm 50 artists for user and the genres they belong to
# user_top_genres = list(pd.core.common.flatten(
#     list(map(lambda artist: longterm_top_artists[artist]['genres'],
#              longterm_top_artists))
# ))