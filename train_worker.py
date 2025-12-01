# Arquivo: train_worker.py

import preditor
import time

# A lista de todas as moedas que queremos treinar
MOEDAS = [
    "BTC/USDT", "ETH/USDT", "XRP/USDT", "LTC/USDT", "BCH/USDT", 
    "ADA/USDT", "DOGE/USDT", "DOT/USDT", "UNI/USDT", "LINK/USDT", 
    "SOL/USDT", "MATIC/USDT", "ETC/USDT", "FIL/USDT", "AAVE/USDT", 
    "MKR/USDT", "COMP/USDT", "SNX/USDT", "YFI/USDT", "AVAX/USDT"
]

print(">>> INICIANDO PROCESSO DE TREINAMENTO DE TODOS OS MODELOS &lt;&lt;&lt;")
print(f"Total de modelos a treinar: {len(MOEDAS)}")

start_time_total = time.time()

for i, moeda in enumerate(MOEDAS):
    print("\n" + "="*50)
    print(f"INICIANDO TREINAMENTO PARA: {moeda} ({i+1}/{len(MOEDAS)})")
    print("="*50)
    
    start_time_moeda = time.time()
    
    try:
        # A flag allow_training=True diz ao preditor para executar o treinamento
        resultado = preditor.run_prediction(moeda, allow_training=True)
        if "error" in resultado:
            print(f"!!! ERRO ao treinar {moeda}: {resultado['error']}")
        else:
            print(f"--- SUCESSO ao treinar {moeda} ---")
            
    except Exception as e:
        print(f"!!! EXCECAO GRAVE ao treinar {moeda}: {e}")

    end_time_moeda = time.time()
    print(f"Tempo de treinamento para {moeda}: {round((end_time_moeda - start_time_moeda) / 60, 2)} minutos.")


end_time_total = time.time()
print("\n" + "#"*50)
print("PROCESSO DE TREINAMENTO DE TODOS OS MODELOS CONCLUIDO!")
print(f"Tempo total de execucao: {round((end_time_total - start_time_total) / 60, 2)} minutos.")
print("#"*50)