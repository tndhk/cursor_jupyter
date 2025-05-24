import requests, pandas as pd

# ---- 1. World Bank 年次データ ----
def wb(series, iso):
    url = f"https://api.worldbank.org/v2/country/{iso}/indicator/{series}?format=json&per_page=2000"
    data = requests.get(url).json()[1]
    df = pd.DataFrame([{ 'year': int(r['date']), f"{iso}_{series}": float(r['value']) if r['value'] else None}
                       for r in data if 2000 <= int(r['date']) <= 2024])
    return df.set_index('year')

series_list = ["NY.GDP.MKTP.KD.ZG", "SL.UEM.TOTL.ZS", "NE.CON.PRVT.CD"]
iso_list    = ['CHN', 'KOR']
macro = pd.concat([wb(s, i) for s in series_list for i in iso_list], axis=1)

# ---- 2. FRED 月次レート → 年次平均 ----
import fredapi as fa
fred = fa.Fred(api_key='d8caff505e4c9c16b8d3e81888e0b54c')
fx_series = {'DEXJPUS':'JPYUSD', 'DEXCHUS':'CNYUSD', 'DEXKOUS':'KRWUSD'}
fx = pd.concat({name: fred.get_series(sid, observation_start='2000-01-01')
                for sid, name in fx_series.items()}, axis=1)

# 円クロス計算
fx['CNYJPY'] = fx['CNYUSD'] / fx['JPYUSD']
fx['KRWJPY'] = fx['KRWUSD'] / fx['JPYUSD']

annual_fx = fx.resample('A').mean().loc['2000':'2024']
annual_fx.index = annual_fx.index.year

# ---- 3. 結合 & 保存 ----
panel = macro.join(annual_fx)
panel.to_csv('china_korea_2000_2024.csv')
print("Saved china_korea_2000_2024.csv")
