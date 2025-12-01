# ==============================================================================
#      ACUMULADOR DIARIO DE DADOS
# ==============================================================================
# Este script busca novos dados e os anexa a um arquivo de historico.

import os, sys, time, datetime
import ccxt, pandas as pd

def acumular_dados(symbol):
    DATA_FOLDER = "Historico_Moedas"
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    file_path = os.path.join(DATA_FOLDER, f'historico_{symbol.replace("/", "_")}.csv')
    exchange = ccxt.binance()
    
    df_existente = pd.DataFrame()
    since = None

    if os.path.exists(file_path):
        print(f"Arquivo de historico encontrado para {symbol}. Lendo o ultimo registro...")
        df_existente = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
        
        if not df_existente.empty:
            last_date = df_existente.index.max()
            since = int(last_date.timestamp() * 1000)
            print(f"  - Ultima data registrada: {last_date.date()}")
    
    if since is None:
        print(f"Nenhum historico encontrado para {symbol}. Baixando o historico completo inicial...")
        # Baixa o historico completo na primeira vez
        limit_dias = 5000
        ohlcv = []
        num_fetches = (limit_dias // 1000) + 1
        temp_since = exchange.milliseconds() - (limit_dias * 24 * 60 * 60 * 1000)
        for _ in range(num_fetches):
            try:
                temp_ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=temp_since, limit=1000)
                if not temp_ohlcv: break
                ohlcv.extend(temp_ohlcv)
                temp_since = temp_ohlcv[-1][0] + 1
            except Exception as e:
                print(f"  - Erro ao buscar lote inicial: {e}"); time.sleep(2)
    else:
        print(f"Buscando novos dados para {symbol} desde a ultima data...")
        # Baixa apenas os dados mais recentes
        ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=since, limit=2000) # Limite alto para garantir que pegamos tudo em caso de inatividade

    if not ohlcv:
        print(f"Nenhum dado novo encontrado para {symbol}.")
        return

    df_novo = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_novo['timestamp'] = pd.to_datetime(df_novo['timestamp'], unit='ms')
    df_novo.set_index('timestamp', inplace=True)
    df_novo.drop_duplicates(keep='first', inplace=True)

    df_final = pd.concat([df_existente, df_novo])
    df_final = df_final[~df_final.index.duplicated(keep='last')] # Remove duplicatas caso haja sobreposicao
    df_final.sort_index(inplace=True)
    
    df_final.to_csv(file_path)
    print(f"Historico para {symbol} atualizado com sucesso! Total de {len(df_final)} registros.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_symbol = sys.argv[1].upper()
        acumular_dados(target_symbol)
    else:
        print("Forneca o simbolo da moeda.")