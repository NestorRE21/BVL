# -*- coding: utf-8 -*-
"""
bvl_data.py — Conexión a la API de la BVL (Bolsa de Valores de Lima)
=====================================================================
Reemplaza yfinance como fuente de datos.

Flujo:
  1. get_token()  → OAuth2 Client Credentials → access_token
  2. get_history(ticker, start, end, token) → precios históricos

Las credenciales se leen de st.secrets (nunca hardcodeadas).
"""
from __future__ import annotations
import requests
import pandas as pd
import numpy as np

# Endpoints
TOKEN_URL = "https://auth-bvl-prod.bvl.com.pe/oauth2/token"
DATA_URL  = "https://api.bvl.com.pe/core/v1/historical-stock-quotes"


def get_token(client_id: str, client_secret: str) -> str | None:
    """
    Obtiene un access_token vía OAuth2 Client Credentials.
    Intenta dos métodos: credenciales en el body (según doc BVL) y,
    si falla, Basic Auth en el header (fallback estándar OAuth2).
    """
    # Método 1: credenciales en el body (lo que dice la doc de la BVL)
    try:
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("access_token")
    except Exception as e:
        print(f"Método body falló: {e}")

    # Método 2: Basic Auth en header (fallback)
    try:
        resp = requests.post(
            TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("access_token")
        print(f"Método basic-auth devolvió {resp.status_code}")
    except Exception as e:
        print(f"Método basic-auth falló: {e}")

    return None


def get_history(ticker: str, start: str, end: str,
                token: str, api_key: str) -> pd.Series | None:
    """
    Descarga precios de cierre históricos de un ticker.
    start/end en formato 'YYYYMMDD' (ej: '20210707').
    Devuelve una Serie de precios de cierre indexada por fecha, o None.
    """
    try:
        resp = requests.get(
            DATA_URL,
            params={"start-date": start, "end-date": end, "ticker": ticker},
            headers={
                "x-api-key": api_key,
                "Authorization": f"Bearer {token}",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        body = data.get("body", [])
        if not body:
            return None
        df = pd.DataFrame(body)
        if "date" not in df.columns or "close" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df[df["close"] > 0].dropna(subset=["close"])
        s = df.set_index("date")["close"].sort_index()
        s.name = ticker
        # Eliminar duplicados de fecha (quedarse con el último)
        s = s[~s.index.duplicated(keep="last")]
        return s
    except Exception as e:
        print(f"Error descargando {ticker}: {e}")
        return None


def download_prices(tickers, start, end, client_id, client_secret, api_key):
    """
    Descarga precios de varios tickers y devuelve un DataFrame de log-retornos
    (fechas × tickers), replicando el formato que la app espera de yfinance.
    """
    token = get_token(client_id, client_secret)
    if token is None:
        return None

    series = {}
    for tk in tickers:
        s = get_history(tk, start, end, token, api_key)
        if s is not None and len(s) > 5:
            series[tk] = s

    if not series:
        return None

    # Combinar en DataFrame de precios, alinear por fecha
    prices = pd.DataFrame(series)
    # Rellenar días sin cotización (feriados/baja liquidez) con forward-fill
    prices = prices.sort_index().ffill()
    # Log-retornos
    log_ret = np.log(prices / prices.shift(1))
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return log_ret
