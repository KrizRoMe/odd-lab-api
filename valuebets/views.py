# valuebets/views.py
from __future__ import annotations
from typing import Optional, Dict, List, Any, Tuple
from math import isfinite
import datetime as dt
import requests
from scipy.stats import poisson

from django.http import JsonResponse
from django.views import View

# ---------------------------
# Configs
# ---------------------------
API_KEY = "f27cce80c21e1a0b0e2cdf2c5ac01f12"
HEADERS = {"x-apisports-key": API_KEY}
BOOKMAKER_BET365 = 8
BET_OVERUNDER = 5

EDGE_MIN_PCT = 2.0
EDGE_MAX_PCT = 15.0
MAX_KELLY_FRACTION = 0.5
MAX_STAKE_PCT = 0.05
MIN_MATCHES_FOR_SEASON = 10
SHRINKAGE_K = 15
HOME_ADVANTAGE = 1.05
MAX_VALUEBETS = 30

# ---------------------------
# Services: API-Football
# ---------------------------
class FootballAPIService:
    """Servicio para consumir API-Football"""

    @staticmethod
    def fetch_all_odds_for_date(date_str: str) -> List[Dict[str, Any]]:
        url = "https://v3.football.api-sports.io/odds"
        page, all_items = 1, []
        while True:
            params = {"date": date_str, "bookmaker": BOOKMAKER_BET365, "bet": BET_OVERUNDER, "page": page}
            resp = requests.get(url, headers=HEADERS, params=params)
            resp.raise_for_status()
            data = resp.json()
            all_items.extend(data.get("response", []))
            paging = data.get("paging", {})
            if not paging or page >= paging.get("total", 1):
                break
            page += 1
        return all_items

    @staticmethod
    def fetch_fixture_by_id(fixture_id: int) -> Optional[Dict[str, Any]]:
        url = "https://v3.football.api-sports.io/fixtures"
        resp = requests.get(url, headers=HEADERS, params={"id": fixture_id})
        resp.raise_for_status()
        arr = resp.json().get("response") or []
        return arr[0] if arr else None

    @staticmethod
    def fetch_team_statistics(league_id: int, team_id: int, season: str) -> Optional[Dict[str, Any]]:
        url = "https://v3.football.api-sports.io/teams/statistics"
        resp = requests.get(url, headers=HEADERS, params={"league": league_id, "team": team_id, "season": season})
        resp.raise_for_status()
        return resp.json().get("response")

    @staticmethod
    def fetch_team_statistics_for_seasons(league_id: int, team_id: int, seasons: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        stats = {}
        for season in seasons:
            try:
                stats[season] = FootballAPIService.fetch_team_statistics(league_id, team_id, season)
            except Exception:
                stats[season] = None
        return stats

# ---------------------------
# Services: Probabilidades / Poisson / Kelly
# ---------------------------
class BettingCalculator:
    @staticmethod
    def safe_float(x: Any) -> Optional[float]:
        try:
            return float(x) if x is not None else None
        except Exception:
            return None

    @staticmethod
    def extract_team_goals_averages(team_stats: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not team_stats:
            return None
        goals = team_stats.get("goals", {})
        avg = goals.get("for", {}).get("average", {})
        against_avg = goals.get("against", {}).get("average", {})
        fixtures = team_stats.get("fixtures", {}).get("played", {})
        mh, ma = int(fixtures.get("home") or 0), int(fixtures.get("away") or 0)
        mt = int(fixtures.get("total") or (mh + ma))
        return {
            "avg_for_home": BettingCalculator.safe_float(avg.get("home")),
            "avg_for_away": BettingCalculator.safe_float(avg.get("away")),
            "avg_for_total": BettingCalculator.safe_float(avg.get("total")),
            "avg_against_home": BettingCalculator.safe_float(against_avg.get("home")),
            "avg_against_away": BettingCalculator.safe_float(against_avg.get("away")),
            "avg_against_total": BettingCalculator.safe_float(against_avg.get("total")),
            "matches_home": mh,
            "matches_away": ma,
            "matches_total": mt,
        }

    @staticmethod
    def extract_league_averages(team_stats: Optional[Dict[str, Any]]) -> Dict[str, float]:
        if not team_stats:
            return {"league_avg_home": 1.3, "league_avg_away": 1.0, "league_avg_total": 2.3}

        goals = team_stats.get("goals", {})
        avg = goals.get("for", {}).get("average", {})

        return {
            "league_avg_home": float(avg.get("home") or 1.3),
            "league_avg_away": float(avg.get("away") or 1.0),
            "league_avg_total": float(avg.get("total") or 2.3)
        }

    @staticmethod
    def compute_expected_goals_combined(home_stats_seasons: Dict[str, Optional[Dict[str, Any]]],
                                        away_stats_seasons: Dict[str, Optional[Dict[str, Any]]]) -> float:
        seasons = sorted(list(set(home_stats_seasons.keys()) | set(away_stats_seasons.keys())), reverse=True)
        total_weight = exp_home_acc = exp_away_acc = 0
        for s in seasons:
            hs = BettingCalculator.extract_team_goals_averages(home_stats_seasons.get(s))
            ats = BettingCalculator.extract_team_goals_averages(away_stats_seasons.get(s))
            if not hs and not ats:
                continue
            league_avg = BettingCalculator.extract_league_averages(hs or ats)
            n_matches = max(hs["matches_total"] if hs else 0, ats["matches_total"] if ats else 0)
            if n_matches == 0:
                continue
            lav_home, lav_away = league_avg["league_avg_home"], league_avg["league_avg_away"]
            home_attack = (hs["avg_for_home"] / lav_home) if hs and hs["avg_for_home"] else 1.0
            home_defense = (hs["avg_against_home"] / lav_home) if hs and hs["avg_against_home"] else 1.0
            away_attack = (ats["avg_for_away"] / lav_away) if ats and ats["avg_for_away"] else 1.0
            away_defense = (ats["avg_against_away"] / lav_away) if ats and ats["avg_against_away"] else 1.0

            exp_home_s = lav_home * home_attack * away_defense * HOME_ADVANTAGE
            exp_away_s = lav_away * away_attack * home_defense / HOME_ADVANTAGE
            w = n_matches / (n_matches + SHRINKAGE_K)
            prior_home, prior_away = lav_home * HOME_ADVANTAGE, lav_away / HOME_ADVANTAGE
            exp_home = w * exp_home_s + (1 - w) * prior_home
            exp_away = w * exp_away_s + (1 - w) * prior_away

            exp_home_acc += exp_home * n_matches
            exp_away_acc += exp_away * n_matches
            total_weight += n_matches

        if total_weight == 0:
            la = BettingCalculator.extract_league_averages(None)
            return la["league_avg_home"] * HOME_ADVANTAGE + la["league_avg_away"] / HOME_ADVANTAGE
        return (exp_home_acc + exp_away_acc) / total_weight

    @staticmethod
    def p_under_over(lambda_total: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        if lambda_total is None:
            return None, None
        p_under = float(sum(poisson.pmf(k, float(lambda_total)) for k in range(3)))
        return float(1 - p_under), p_under

    @staticmethod
    def implied_probs_from_odds(odd_over: Optional[float], odd_under: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        if odd_over is None or odd_under is None:
            return None, None
        ro, ru = 1 / odd_over, 1 / odd_under
        s = ro + ru
        return (ro / s, ru / s) if s > 0 else (None, None)

    @staticmethod
    def kelly_fraction(p: float, decimal_odds: float) -> float:
        b = decimal_odds - 1.0
        if b <= 0: return 0.0
        f = (p * decimal_odds - 1.0) / b
        if not isfinite(f) or f <= 0: return 0.0
        return min(f, MAX_KELLY_FRACTION)

# ---------------------------
# View: Class Based
# ---------------------------
class ValueBetsView(View):
    """Vista para calcular ValueBets"""

    def get(self, request, *args, **kwargs) -> JsonResponse:
        tomorrow = (dt.date.today() + dt.timedelta(days=1)).strftime("%Y-%m-%d")
        bankroll: float = float(request.GET.get("bankroll", 100.0) or 100.0)

        try:
            odds_items = FootballAPIService.fetch_all_odds_for_date(tomorrow)
        except Exception as e:
            return JsonResponse({"error": "API-Football error fetching odds", "details": str(e)}, status=502)

        fixture_cache: Dict[int, Optional[Dict[str, Any]]] = {}
        results: List[Dict[str, Any]] = []

        for item in odds_items:
            if len(results) >= MAX_VALUEBETS:
                break

            fixture_info = item.get("fixture") or {}
            fixture_id = fixture_info.get("id")
            league_info = item.get("league") or {}
            league_id, season = league_info.get("id"), league_info.get("season")

            # Cache fixture details
            if fixture_id in fixture_cache:
                detail = fixture_cache[fixture_id]
            else:
                if fixture_id is None:
                    continue

                fixture_id_int = int(fixture_id)  # forzar a int
                detail = FootballAPIService.fetch_fixture_by_id(fixture_id_int)
                fixture_cache[fixture_id_int] = detail
            if not detail:
                continue

            teams = detail.get("teams", {})
            home, away = teams.get("home", {}), teams.get("away", {})
            home_id, away_id = home.get("id"), away.get("id")

            odd_over, odd_under = self._extract_odds(item)
            if odd_over is None and odd_under is None:
                continue

            # Obtener stats combinando temporadas
            seasons_to_try = [str(season or "2025"), "2024"]

            if league_id is None or home_id is None or away_id is None:
                continue  # saltar este partido

            home_stats = FootballAPIService.fetch_team_statistics_for_seasons(league_id, home_id, seasons_to_try)
            away_stats = FootballAPIService.fetch_team_statistics_for_seasons(league_id, away_id, seasons_to_try)

            lam = BettingCalculator.compute_expected_goals_combined(home_stats, away_stats)
            p_over, p_under = BettingCalculator.p_under_over(lam)
            implied_over, implied_under = BettingCalculator.implied_probs_from_odds(odd_over, odd_under)

            tip = self._calculate_tip(bankroll, p_over, p_under, implied_over, implied_under, odd_over, odd_under)
            if tip:
                results.append({
                    "fixture_id": fixture_id,
                    "league": league_info.get("name"),
                    "date": fixture_info.get("date"),
                    "home": {"id": home_id, "name": home.get("name")},
                    "away": {"id": away_id, "name": away.get("name")},
                    "odd_over": odd_over,
                    "odd_under": odd_under,
                    "model_lambda": round(lam, 3) if lam else None,
                    "p_over_model": round(p_over, 4) if p_over else None,
                    "p_under_model": round(p_under, 4) if p_under else None,
                    "tip": tip
                })

        return JsonResponse({"date": tomorrow, "bankroll": bankroll, "count": len(results), "results": results}, safe=False)

    def _extract_odds(self, item: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        odd_over = odd_under = None
        try:
            for bm in item.get("bookmakers", []):
                if bm.get("id") == BOOKMAKER_BET365:
                    for bet in bm.get("bets", []):
                        if bet.get("id") == BET_OVERUNDER:
                            for v in bet.get("values", []):
                                val, odd = v.get("value"), v.get("odd")
                                if val and odd:
                                    low = val.strip().lower()
                                    if low == "over 2.5":
                                        odd_over = float(odd)
                                    elif low == "under 2.5":
                                        odd_under = float(odd)
        except Exception:
            pass
        return odd_over, odd_under

    def _calculate_tip(
        self,
        bankroll: float,
        p_over: Optional[float],
        p_under: Optional[float],
        implied_over: Optional[float],
        implied_under: Optional[float],
        odd_over: Optional[float],
        odd_under: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        tip: Optional[Dict[str, Any]] = None
        # Over
        if p_over and odd_over and implied_over is not None:
            edge_over_pct = (p_over - implied_over) * 100
            if EDGE_MIN_PCT <= edge_over_pct <= EDGE_MAX_PCT:
                f = BettingCalculator.kelly_fraction(p_over, odd_over)
                stake = round(bankroll * min(f, MAX_STAKE_PCT), 2)
                tip = {
                    "market": "Over 2.5",
                    "model_prob": round(p_over, 4),
                    "implied_prob": round(implied_over, 4),
                    "edge_pct": round(edge_over_pct, 3),
                    "kelly_fraction": round(f, 4),
                    "suggested_stake": stake
                }

        # Under
        if tip is None and p_under and odd_under and implied_under is not None:
            edge_under_pct = (p_under - implied_under) * 100
            if EDGE_MIN_PCT <= edge_under_pct <= EDGE_MAX_PCT:
                f = BettingCalculator.kelly_fraction(p_under, odd_under)
                stake = round(bankroll * min(f, MAX_STAKE_PCT), 2)
                tip = {
                    "market": "Under 2.5",
                    "model_prob": round(p_under, 4),
                    "implied_prob": round(implied_under, 4),
                    "edge_pct": round(edge_under_pct, 3),
                    "kelly_fraction": round(f, 4),
                    "suggested_stake": stake
                }
        return tip
