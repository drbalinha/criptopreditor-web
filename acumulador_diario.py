import os, sys, time
import ccxt
import pandas as pd

def acumular_dados(symbol):
    DATA_FOLDER = "Historico_Moedas"
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    file_path = os.path.join(DATA_FOLDER, f'historico_{symbol.replace("/", "_")}.csv')
    exchange = ccxt.binance()
    
    if os.path.exists(file_path):
        df_existente = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
        if not df_existente.empty:
            since = int(df_existente.index.max().timestamp() * 1000)
        else:
            since = None
    else:
        since = None

    if since is None:
        ohlcv = []
        limit_dias = 1500
        temp_since = exchange.milliseconds() - (limit_dias * 24 * 60 * 60 * 1000)

        while True:
            lote = exchange.fetch_ohlcv(symbol, '1d', since=temp_since, limit=1000)
            if not lote:
                break
            ohlcv.extend(lote)
            temp_since = lote[-1][0] + 1
    else:
        ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=since, limit=1500)

    df_novo = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_novo['timestamp'] = pd.to_datetime(df_novo['timestamp'], unit='ms')
    df_novo.set_index('timestamp', inplace=True)

    if os.path.exists(file_path):
        df_final = pd.concat([df_existente, df_novo])
    else:
        df_final = df_novo

    df_final = df_final[~df_final.index.duplicated(keep='last')]
    df_final.sort_index(inplace=True)
    df_final.to_csv(file_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Forneça o símbolo.")
        sys.exit(1)

    acumular_dados(sys.argv[1].upper())
