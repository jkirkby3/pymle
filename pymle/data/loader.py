import pandas as pd
import pathlib


def _get_fpath(fname: str) -> str:
    return str(pathlib.Path(__file__).parent.joinpath(fname).resolve())


def load_VIX() -> pd.DataFrame:
    """ Load VIX data history """
    return pd.read_csv(_get_fpath("VIX.csv"))


def load_FX_USD_EUR() -> pd.DataFrame:
    """ Load EUR / USD data history"""
    return pd.read_csv(_get_fpath("FX_EUR_USD.csv"))


def load_10yr_CMrate() -> pd.DataFrame:
    return pd.read_csv(_get_fpath("10yrCMrate.csv"))

