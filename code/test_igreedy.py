"""
Tests for the iGreedy anycast geolocation algorithm.

Three test cases with synthetic latency measurements:
  Case 1: Unicast — single 1ms measurement at Amsterdam → AMX
  Case 2: Three-site anycast — one 1ms VP per site (AMX, BFI, JNB)
  Case 3: Three-site anycast — multiple overlapping VPs per site

Airport coordinates (from the embedded dataset):
  AMX  Amsterdam      52.3086,   4.7639   (pop 741636)
  BFI  Seattle        47.5300, -122.3020  (pop 608660)
  JNB  Johannesburg  -26.1392,  28.2460   (pop 4434827)
  QRA  Johannesburg  -26.2425,  28.1512   (pop 4434827)
  HLA  Johannesburg  -25.9385,  27.9261   (pop 4434827)

Nearby VPs used in case 3:
  LHR  London         51.4706,  -0.4619   (~355 km from AMX)
  BRU  Brussels       50.9014,   4.4844   (~186 km from AMX)
  PDX  Portland       45.5887, -122.5980  (~215 km from BFI)
  YVR  Vancouver      49.1939, -123.1840  (~196 km from BFI)
  DUR  Durban        -29.6144,  31.1197   (~398 km from JNB)
  CPT  Cape Town     -33.9648,  18.6017   (~1263 km from JNB)

Radius formula: radius_km = rtt_ms * 0.001 * 299792.458 / 1.52 / 2  ≈  rtt_ms * 98.616
  1 ms → ~99 km radius
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd
from math import radians

# Ensure the code directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

from anycast import AnycastDF
from igreedy import get_airports, analyze_df

FIBER_RI = np.float32(1.52)
SPEED_OF_LIGHT = np.float32(299792.458)

# Johannesburg has 3 airports (JNB, QRA, HLA) with identical population;
# the scoring tiebreaker is non-deterministic among them.
JOHANNESBURG_IATAS = {"JNB", "QRA", "HLA"}


# ── helpers ──────────────────────────────────────────────────────────────

def make_measurement(target, hostname, lat, lon, rtt):
    """Create a single-row dict matching the input DataFrame schema."""
    return {
        "target": target,
        "hostname": hostname,
        "lat": np.float32(lat),
        "lon": np.float32(lon),
        "rtt": np.float32(rtt),
        "lat_rad": np.float32(radians(lat)),
        "lon_rad": np.float32(radians(lon)),
        "radius": np.float32(rtt * 0.001 * SPEED_OF_LIGHT / FIBER_RI / 2.0),
    }


def make_df(rows):
    """Build a DataFrame from a list of measurement dicts."""
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def airports_df():
    """Load the real airports dataset once for all tests."""
    return get_airports()


# ── Case 1: unicast ─────────────────────────────────────────────────────

class TestCase1Unicast:
    """Single 1ms measurement centred on Amsterdam → MIS=1, geolocates to AMX."""

    @staticmethod
    def _discs():
        return make_df([
            make_measurement("1.2.3.4", "vp-ams", 52.3086, 4.7639, 1.0),
        ])

    def test_enumeration(self, airports_df):
        in_df = self._discs()
        anycast = AnycastDF(in_df, airports_df, alpha=1.0)
        num_sites, mis_df = anycast.enumeration()
        assert num_sites == 1, "single disc ⇒ MIS size must be 1"

    def test_geolocation(self, airports_df):
        in_df = self._discs()
        in_df['processed'] = False  # required by analyze_df
        result = analyze_df(in_df, alpha=1.0, airports_df=airports_df, anycast_only=False)
        assert result is not None, "unicast must produce a result"
        assert len(result) == 1, "unicast must produce exactly 1 result"
        assert result.iloc[0]["pop_iata"] == "AMX", "must geolocate to Amsterdam (AMX)"

    def test_skipped_with_anycast_flag(self, airports_df):
        in_df = self._discs()
        in_df['processed'] = False
        result = analyze_df(in_df, alpha=1.0, airports_df=airports_df, anycast_only=True)
        assert result is None, "--anycast flag must skip unicast targets"


# ── Case 2: three-site anycast (one VP per site, 1ms each) ──────────────

class TestCase2ThreeSites:
    """Three 1ms discs on AMX, BFI, JNB — far apart → MIS=3."""

    @staticmethod
    def _discs():
        return make_df([
            make_measurement("9.9.9.9", "vp-ams",  52.3086,    4.7639, 1.0),
            make_measurement("9.9.9.9", "vp-sea",  47.5300, -122.3020, 1.0),
            make_measurement("9.9.9.9", "vp-jnb", -26.1392,   28.2460, 1.0),
        ])

    def test_enumeration(self, airports_df):
        in_df = self._discs()
        anycast = AnycastDF(in_df, airports_df, alpha=1.0)
        num_sites, mis_df = anycast.enumeration()
        assert num_sites == 3, "three far-apart 1ms discs ⇒ MIS size must be 3"

    def test_geolocation(self, airports_df):
        in_df = self._discs()
        result = analyze_df(in_df, alpha=1.0, airports_df=airports_df)
        assert result is not None
        assert len(result) == 3, "must produce exactly 3 results"
        iatas = set(result["pop_iata"])
        assert "AMX" in iatas, "must contain Amsterdam (AMX)"
        assert "BFI" in iatas, "must contain Seattle (BFI)"
        assert iatas & JOHANNESBURG_IATAS, "must contain a Johannesburg airport"


# ── Case 3: three-site anycast with multiple overlapping VPs ─────────────

class TestCase3MultiVP:
    """
    Each site has 3 VPs whose discs overlap locally but NOT across sites.

    Amsterdam cluster:
      VP at AMX   (1 ms, r ≈  99 km)
      VP at BRU   (3 ms, r ≈ 296 km)
      VP at LHR   (5 ms, r ≈ 493 km)

    Seattle cluster:
      VP at BFI   (1 ms, r ≈  99 km)
      VP at PDX   (4 ms, r ≈ 394 km)
      VP at YVR   (4 ms, r ≈ 394 km)

    Johannesburg cluster:
      VP at JNB   (1 ms, r ≈  99 km)
      VP at DUR   (6 ms, r ≈ 592 km)
      VP at CPT  (15 ms, r ≈1479 km)
    """

    @staticmethod
    def _discs():
        return make_df([
            # Amsterdam cluster
            make_measurement("8.8.8.8", "vp-ams",  52.3086,    4.7639,  1.0),
            make_measurement("8.8.8.8", "vp-bru",  50.9014,    4.4844,  3.0),
            make_measurement("8.8.8.8", "vp-lhr",  51.4706,   -0.4619,  5.0),
            # Seattle cluster
            make_measurement("8.8.8.8", "vp-sea",  47.5300, -122.3020,  1.0),
            make_measurement("8.8.8.8", "vp-pdx",  45.5887, -122.5980,  4.0),
            make_measurement("8.8.8.8", "vp-yvr",  49.1939, -123.1840,  4.0),
            # Johannesburg cluster
            make_measurement("8.8.8.8", "vp-jnb", -26.1392,   28.2460,  1.0),
            make_measurement("8.8.8.8", "vp-dur", -29.6144,   31.1197,  6.0),
            make_measurement("8.8.8.8", "vp-cpt", -33.9648,   18.6017, 15.0),
        ])

    def test_enumeration(self, airports_df):
        in_df = self._discs()
        anycast = AnycastDF(in_df, airports_df, alpha=1.0)
        num_sites, mis_df = anycast.enumeration()
        assert num_sites == 3, "three clusters of overlapping discs ⇒ MIS size must be 3"

    def test_geolocation(self, airports_df):
        in_df = self._discs()
        result = analyze_df(in_df, alpha=1.0, airports_df=airports_df)
        assert result is not None
        assert len(result) == 3, "must produce exactly 3 results"
        iatas = set(result["pop_iata"])
        assert "AMX" in iatas, "must contain Amsterdam (AMX)"
        assert "BFI" in iatas, "must contain Seattle (BFI)"
        assert iatas & JOHANNESBURG_IATAS, "must contain a Johannesburg airport"

    def test_no_phantom_sites(self, airports_df):
        """Every result must be one of the expected airports — no phantom sites."""
        in_df = self._discs()
        result = analyze_df(in_df, alpha=1.0, airports_df=airports_df)
        assert result is not None
        valid = {"AMX", "BFI"} | JOHANNESBURG_IATAS
        for iata in result["pop_iata"]:
            assert iata in valid, f"unexpected airport {iata} in results"
