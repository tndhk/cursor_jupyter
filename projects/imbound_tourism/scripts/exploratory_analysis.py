# %% [markdown]
# # 訪問者数の推移
# ### 1. データの読み込みと前処理
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
try:
    fm._rebuild()
except AttributeError:
    # _rebuild が見つからない古いバージョンの matplotlib の場合
    # あるいは別の方法を試す (今回は何もしない)
    pass
import seaborn as sns

# 利用可能なフォントを表示
available_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
print('Available fonts:', available_fonts)

# フォントに関するエラーメッセージを抑制
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# グラフの日本語表示設定
# 環境に合わせて適宜変更してください (Mac, Windows, Linuxの例)
# plt.rcParams['font.sans-serif'] = ['Hiragino Sans']  # 利用可能なフォントを指定
# plt.rcParams['font.sans-serif'] = ['MS Gothic'] # Windowsの場合
# plt.rcParams['font.family'] = 'IPAexGothic' # LinuxなどでIPAフォントをインストールした場合
# plt.rcParams['axes.unicode_minus'] = False # マイナス記号の表示設定

# matplotlibのデフォルト設定を使用
plt.rcdefaults()

# %%
df_raw = pd.read_csv('../data/visitors.csv')
print("--- df_raw.head() ---")
print(df_raw.head())
print("\n--- df_raw.info() ---")
df_raw.info()

# データには注釈行や国名が空欄の行、集計行が含まれているため、これらを処理します。
# 注釈行の削除 (country列が '注１' または '注２' で始まる行を除外)

df_cleaned = df_raw[~df_raw['country'].str.startswith('注１', na=False) & ~df_raw['country'].str.startswith('注２', na=False)].copy()

# visitors列を数値型に変換 (変換できないものはNaNに)
df_cleaned.loc[:, 'visitors'] = pd.to_numeric(df_cleaned['visitors'], errors='coerce')

# 国名がNaNまたは空欄の行を除外
df_cleaned.dropna(subset=['country'], inplace=True)
df_cleaned = df_cleaned[df_cleaned['country'].str.strip() != '']

# 集計行を除外 (例: '総数', '計' で終わるもの)
excluded_keywords = ['総数', '計', 'その他'] # 'その他' も広すぎるため一旦除外検討
pattern = '|'.join(excluded_keywords) # '総数' OR '計' OR 'その他'
df_cleaned = df_cleaned[~df_cleaned['country'].str.contains(pattern, na=False)]

print("\n--- df_cleaned.head() after exclusions ---")
print(df_cleaned.head())
print(f"除外処理後のデータ数: {len(df_cleaned)}")

# 次に、`year` と `month` から日付型の列を作成します。
# 月の表記が「1月」などになっているため、数値に変換します。

def convert_month_to_int(month_str):
    if isinstance(month_str, str):
        return int(month_str.replace('月', ''))
    return np.nan

df_cleaned.loc[:, 'month_num'] = df_cleaned['month'].apply(convert_month_to_int)

# year, month_num, day=1としてdatetimeオブジェクトを作成 (dayは仮に1日とする)
# 欠損値を含む行はここで除外されるか、NaTになる
df_cleaned.dropna(subset=['year', 'month_num'], inplace=True) # yearやmonth_numがNaNの行を除外
df_cleaned.loc[:, 'date'] = pd.to_datetime(df_cleaned['year'].astype(int).astype(str) + '-' +
                                     df_cleaned['month_num'].astype(int).astype(str) + '-' + '1', errors='coerce')

# 不要になったmonth_num列を削除、visitorsがNaNの行もこの段階で除去
df_cleaned.drop(columns=['month_num'], inplace=True)
df_cleaned.dropna(subset=['visitors', 'date'], inplace=True) # visitors や date がNaNの行を除外
df_cleaned.loc[:, 'visitors'] = df_cleaned['visitors'].astype(int) # visitorsを整数型に

print("\n--- df_cleaned.head() after date conversion ---")
print(df_cleaned.head())
print("\n--- df_cleaned.info() after date conversion ---")
df_cleaned.info()

# 2025年のデータを除外 (まだ集計中の年のため)
df_cleaned = df_cleaned[df_cleaned['year'] != 2025].copy()

# %% [markdown]
# ### 2. 年間訪問者数と国別年間訪問者数の集計
# - 年間の総訪問者数と、国別の年間の訪問者数を集計します。

# %%
# 年間総訪問者数の集計
df_yearly_total = df_cleaned.groupby('year')['visitors'].sum().reset_index()
print("\n--- 年間総訪問者数 ---")
print(df_yearly_total)

# 国別年間訪問者数の集計
df_yearly_country = df_cleaned.groupby(['year', 'country'])['visitors'].sum().reset_index()
print("\n--- 国別年間訪問者数 (一部) ---") # 全て表示すると長くなるため一部のみ
print(df_yearly_country.head())

# %% [markdown]
# ### 3. データ期間全体での国別訪問者数合計 TOP 10
# - これまでの全期間で、訪問者数が多かった国の上位10カ国を集計します。

# %%
# データ期間全体での国別訪問者数合計を集計
df_country_total = df_cleaned.groupby('country')['visitors'].sum().reset_index()

# 合計訪問者数で降順にソートし、上位10カ国を取得
df_top_10_countries = df_country_total.sort_values(by='visitors', ascending=False).head(10)

print("\n--- データ期間全体での国別訪問者数合計 TOP 10 ---")
print(df_top_10_countries)

# %% [markdown]
# ### 4. 年ごとの国別訪問者数合計 TOP 10
# - 各年で、訪問者数が多かった国の上位10カ国を集計します。

# %%
# 年ごとの国別訪問者数合計を集計し、各年でTOP 10を取得
print("\n--- 年ごとの国別訪問者数合計 TOP 10 ---")

# 各年について処理
for year in sorted(df_cleaned['year'].unique()):
    print(f"\n--- {year}年 ---")
    print("") # ここに改行を追加
    # 該当年のデータのみを抽出
    df_yearly_data = df_cleaned[df_cleaned['year'] == year].copy()

    # 国別訪問者数合計を集計
    df_country_yearly_total = df_yearly_data.groupby('country')['visitors'].sum().reset_index()

    # 合計訪問者数で降順にソートし、上位10カ国を取得
    df_top_10_yearly = df_country_yearly_total.sort_values(by='visitors', ascending=False).head(10)

    print(df_top_10_yearly)

# %% [markdown]
# ### 5. 主要国の年間訪問者数推移の可視化
# - 年ごとのデータが集計できたので、主要な国の訪問者数推移をグラフで可視化します。

# %%
# 可視化対象とする国リスト (データ期間や年ごとのトップ10を参考に選定)
# 必要に応じて表示したい国を変更してください
target_countries_jp = ['韓国', '中国', '台湾', '米国', 'タイ', '香港', '豪州', 'ベトナム', 'マレーシア', 'フィリピン']

# 日本語の国名を英語にマッピングする辞書
country_name_mapping = {
    '韓国': 'South Korea',
    '中国': 'China',
    '台湾': 'Taiwan',
    'タイ': 'Thailand',
    'アメリカ': 'USA',
    '香港': 'Hong Kong',
    'インド': 'India',
    'インドネシア': 'Indonesia',
    'カナダ': 'Canada',
    'シンガポール': 'Singapore',
    'ドイツ': 'Germany',
    'フィリピン': 'Philippines',
    'フランス': 'France',
    'ベトナム': 'Vietnam',
    'マレーシア': 'Malaysia',
    '米国': 'USA',
    '英国': 'UK',
    '豪州': 'Australia',
    # 必要に応じて他の国名も追加してください
}

# 選択した国のみのデータに絞り込み (日本語名でフィルタリング)
df_plot = df_cleaned[df_cleaned['country'].isin(target_countries_jp)].copy()

# 国名列を英語に変換
df_plot.loc[:, 'country'] = df_plot['country'].map(country_name_mapping)

# 年ごとの国別訪問者数を集計し直し (グラフ用に整形)
df_plot_yearly = df_plot.groupby(['year', 'country'])['visitors'].sum().reset_index()

# seabornを使って折れ線グラフを描画
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_plot_yearly, x='year', y='visitors', hue='country', marker='o')

plt.title('Annual Visitor Trends by Country')
plt.xlabel('Year')
plt.ylabel('Number of Visitors')
plt.grid(True)
plt.xticks(df_plot_yearly['year'].unique()) # 各年のラベルを表示
# plt.legend(title='Country') # 凡例は非表示にするためコメントアウト

# 縦軸の表示をMillion単位にする formatter を設定
ax = plt.gca()
def millions_formatter(x, pos):
    '''100万単位で表示するためのformatter関数'''
    return f'{x/1_000_000:.1f}M'
ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))

# 各線の右端にラベルを付ける
# 各国ごとの最新年のデータを取得し、訪問者数でソート
last_data_points = df_plot_yearly.groupby('country').tail(1).sort_values(by='visitors')

# ラベルの縦位置調整のためのオフセット値を定義
# 必要に応じてこの値を調整してください
vertical_offset = last_data_points['visitors'].max() * 0.02 # 例えば最大値の2%程度のオフセット
last_y = -float('inf')

# 各データポイントに対してラベルを追加
for index, row in last_data_points.iterrows():
    country = row['country']
    year = row['year']
    visitors = row['visitors']

    # 重なりを避けるために縦位置を調整
    current_y = visitors
    if abs(current_y - last_y) < (last_data_points['visitors'].max() * 0.01): # 近すぎる場合はオフセット
         current_y += vertical_offset # 少し上にずらす
         vertical_offset *= -1 # 次は下にずらすためにオフセットの符号を反転
    else:
        vertical_offset = abs(vertical_offset) # 離れている場合はオフセットをリセット（最初の正の値に戻す）

    # ラベルの位置を最後の点の少し右に調整
    ax.text(year + 0.1, current_y, country, va='center') # yearに少し値を足して右にずらす
    last_y = current_y # 現在のy座標を記録

plt.tight_layout()
plt.show()
# %%

# %% [markdown]
# ### 6. 年間総訪問者数推移の可視化
# - 全体（全地域合計）の年間の訪問者数推移をグラフで可視化します。

# %%
# 年間総訪問者数グラフの描画
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_yearly_total, x='year', y='visitors', marker='o')

plt.title('Annual Total Visitor Trends')
plt.xlabel('Year')
plt.ylabel('Number of Visitors')
plt.grid(True)
plt.xticks(df_yearly_total['year'].unique()) # 各年のラベルを表示

# 縦軸の表示をMillion単位にする formatter を設定
ax = plt.gca()
def millions_formatter(x, pos):
    '''100万単位で表示するためのformatter関数'''
    return f'{x/1_000_000:.1f}M'
ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))

plt.tight_layout()
plt.show()

# %%

# %% [markdown]
# ### 7. 国別年間訪問者数 成長率の分析
# - 各国の年間訪問者数の前年比成長率を計算し、その推移を可視化します。

# %%
# 国別年間成長率を計算 (デフォルトは前年比)
df_yearly_country['visitors_growth_rate'] = df_yearly_country.groupby('country')['visitors'].pct_change() * 100

# 2024年の成長率を2019年比で再計算
# 2019年の訪問者数を取得 (すべての国に対して)
visitors_2019 = df_yearly_country[df_yearly_country['year'] == 2019][['country', 'visitors']].set_index('country')

# 2024年のデータを抽出し、2019年のデータがある国について成長率を計算
# locを使って直接元のdf_yearly_countryの値を更新

# df_yearly_countryの2024年の行を選択
df_2024_rows = df_yearly_country['year'] == 2024

# 2024年のデータがある各国について処理
for country in df_yearly_country.loc[df_2024_rows, 'country'].unique():
    if country in visitors_2019.index:
        visitors_2024 = df_yearly_country.loc[(df_yearly_country['year'] == 2024) & (df_yearly_country['country'] == country), 'visitors'].iloc[0]
        visitors_2019_val = visitors_2019.loc[country, 'visitors']

        # 2019年の訪問者数が0でないことを確認して成長率を計算
        if visitors_2019_val != 0:
            growth_rate_2019_base = ((visitors_2024 - visitors_2019_val) / visitors_2019_val) * 100
            # 元のデータフレームの該当する場所を更新
            df_yearly_country.loc[(df_yearly_country['year'] == 2024) & (df_yearly_country['country'] == country), 'visitors_growth_rate'] = growth_rate_2019_base
        else:
            # 2019年が0の場合は成長率をNaNまたは特定の記号とする (今回はNaN)
            df_yearly_country.loc[(df_yearly_country['year'] == 2024) & (df_yearly_country['country'] == country), 'visitors_growth_rate'] = float('nan')


# 初年度のデータは成長率がNaNになるため除外、および成長率が計算できなかった国も除外
# 2024年の更新でNaNになったものもここで除外される
df_growth = df_yearly_country.dropna(subset=['visitors_growth_rate']).copy()

# コロナ禍で特殊な期間（2020年, 2021年, 2022年, 2023年）のデータを除外
years_to_exclude_from_growth = [2020, 2021, 2022, 2023]
df_growth_filtered = df_growth[~df_growth['year'].isin(years_to_exclude_from_growth)].copy()

print("\n--- 国別年間成長率 (一部) ---")
# ここで df_growth_filtered を表示
print(df_growth_filtered.head())

# 可視化対象とする国リスト (セクション5と同じ主要国リストを使用)
# 日本語名リストは既に定義されている target_countries_jp を使用
# 英語名にマッピングするための country_name_mapping も既に定義済み

# 選択した国のみの成長率データに絞り込み (日本語名でフィルタリングし、英語名に変換)
# ここで df_growth_filtered を使用
df_growth_plot = df_growth_filtered[df_growth_filtered['country'].isin(target_countries_jp)].copy()
df_growth_plot.loc[:, 'country'] = df_growth_plot['country'].map(country_name_mapping)

# seabornを使って成長率の折れ線グラフを描画
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_growth_plot, x='year', y='visitors_growth_rate', hue='country', marker='o')

plt.title('Annual Visitor Growth Rate by Country (%)') # 英語タイトル
plt.xlabel('Year') # 英語ラベル
plt.ylabel('Growth Rate (%)') # 英語ラベル
plt.grid(True)
plt.xticks(df_growth_plot['year'].unique()) # 各年のラベルを表示
# plt.legend(title='Country', loc='center left', bbox_to_anchor=(1, 0.5)) # 標準凡例は削除

# 各線の右端にラベルを付ける (成長率グラフ用)
ax = plt.gca() # 現在のAxesを取得

# 各国ごとの最新年の成長率データを取得し、成長率でソート
# コロナ禍の年を除外しているため、最新年が国によって異なる可能性があるが、tail(1)で取得
last_growth_points = df_growth_plot.groupby('country').tail(1).sort_values(by='visitors_growth_rate')

# ラベルの縦位置調整のためのオフセット値を定義
# 必要に応じてこの値を調整してください
# 成長率の範囲に応じてオフセットを計算
growth_rate_range = df_growth_plot['visitors_growth_rate'].max() - df_growth_plot['visitors_growth_rate'].min()
vertical_offset_gr = growth_rate_range * 0.03 # 例: レンジの3%程度のオフセット
last_y_gr = -float('inf')


# 各データポイントに対してラベルを追加
for index, row in last_growth_points.iterrows():
    country = row['country']
    year = row['year']
    growth_rate = row['visitors_growth_rate'] # 成長率データを使用

    # 重なりを避けるために縦位置を調整
    current_y_gr = growth_rate
    # 近傍の閾値を定義 (例: レンジの1%以内)
    proximity_threshold = growth_rate_range * 0.01

    if abs(current_y_gr - last_y_gr) < proximity_threshold:
        current_y_gr += vertical_offset_gr # 少しずらす
        vertical_offset_gr *= -1 # 次は逆方向にずらすためにオフセットの符号を反転
    else:
        vertical_offset_gr = abs(vertical_offset_gr) # 離れている場合はオフセットをリセット（最初の正の値に戻す）


    # ラベルの位置を最後の点の少し右に調整
    # x座標は年の少し右、y座標は成長率の値
    ax.text(year + 0.1, current_y_gr, country, va='center')
    last_y_gr = current_y_gr # 現在の調整済みy座標を記録

plt.tight_layout()
plt.show()

# %%

# %% [markdown]
# ### 8. 韓国と中国の2019年 vs 2024年 訪問者数と成長率比較
# - 年間成長率の分析で特に注目した韓国と中国について、2019年と2024年の実際の訪問者数と、2024年の2019年比成長率を具体的に見ていきます。

# %%
# 比較対象の国 (英語名)
countries_to_compare = ['South Korea', 'China']
comparison_data = []

# 2019年の訪問者数を取得 (visitors_2019は日本語国名がキーになっている可能性があるので注意)
# country_name_mapping を逆引きするか、df_yearly_country から直接取得する
# df_yearly_country の 'country' 列は元々日本語なので、英語名に変換する前の日本語名で検索する必要がある
# または、df_growth_plot（英語名になっている）の元になった df_yearly_country から引く

# 英語名から日本語名への逆マッピングを作成 (必要な国のみ)
# country_name_mapping のキーとバリューを入れ替える
country_name_mapping_inv = {v: k for k, v in country_name_mapping.items()}

for country_en in countries_to_compare:
    country_jp = country_name_mapping_inv.get(country_en)
    if not country_jp:
        print(f"Warning: Japanese name not found for {country_en}")
        continue

    # 2019年の訪問者数
    try:
        visits_2019 = df_yearly_country[
            (df_yearly_country['year'] == 2019) & (df_yearly_country['country'] == country_jp)
        ]['visitors'].iloc[0]
    except IndexError:
        visits_2019 = None # データがない場合

    # 2024年の訪問者数
    try:
        visits_2024 = df_yearly_country[
            (df_yearly_country['year'] == 2024) & (df_yearly_country['country'] == country_jp)
        ]['visitors'].iloc[0]
    except IndexError:
        visits_2024 = None # データがない場合

    # 2024年の2019年比成長率 (df_yearly_countryに計算済み)
    try:
        growth_rate_2024_vs_2019 = df_yearly_country[
            (df_yearly_country['year'] == 2024) & (df_yearly_country['country'] == country_jp)
        ]['visitors_growth_rate'].iloc[0]
    except IndexError:
        growth_rate_2024_vs_2019 = None # データがない場合
    
    comparison_data.append({
        'Country': country_en,
        '2019 Visitors': visits_2019,
        '2024 Visitors': visits_2024,
        '2024 Growth Rate (vs 2019)': growth_rate_2024_vs_2019
    })

df_comparison = pd.DataFrame(comparison_data)
print("\n--- Comparison: South Korea vs China (2019 vs 2024) ---")
print(df_comparison)


# %%

# %% [markdown]
# ### 9. 経済指標・為替レートと訪問者数の関連分析 (韓国・中国)
# - `china_korea_2000_2024.csv` から経済指標と為替レートを読み込み、訪問者数データと結合して関連を分析します。

# %%
# 経済・為替データの読み込み
try:
    df_econ_raw = pd.read_csv('china_korea_2000_2024.csv') # スクリプトと同じディレクトリにあると想定
except FileNotFoundError:
    print("Error: 'china_korea_2000_2024.csv' not found. Make sure get_kor_china_data.py ran successfully and the file is in the 'scripts' directory.")
    df_econ_raw = None

if df_econ_raw is not None:
    print("\n--- Raw Economic Data (head) ---")
    print(df_econ_raw.head())

    # データの整形
    # 'year' 列を数値型に変換 (もし文字列なら)
    df_econ_raw['year'] = pd.to_numeric(df_econ_raw['year'], errors='coerce')
    df_econ_raw.dropna(subset=['year'], inplace=True) # yearがNaNの行は除外
    df_econ_raw['year'] = df_econ_raw['year'].astype(int)


    # 必要な列を選択し、国ごとにデータを準備
    # 中国データ
    df_china_econ = df_econ_raw[['year', 'CHN_NY.GDP.MKTP.KD.ZG', 'CHN_SL.UEM.TOTL.ZS', 'CHN_NE.CON.PRVT.CD', 'CNYJPY']].copy()
    df_china_econ.rename(columns={
        'CHN_NY.GDP.MKTP.KD.ZG': 'GDP_Growth',
        'CHN_SL.UEM.TOTL.ZS': 'Unemployment',
        'CHN_NE.CON.PRVT.CD': 'Private_Consumption',
        'CNYJPY': 'FX_vs_JPY'
    }, inplace=True)
    df_china_econ['country'] = 'China'

    # 韓国データ
    df_korea_econ = df_econ_raw[['year', 'KOR_NY.GDP.MKTP.KD.ZG', 'KOR_SL.UEM.TOTL.ZS', 'KOR_NE.CON.PRVT.CD', 'KRWJPY']].copy()
    df_korea_econ.rename(columns={
        'KOR_NY.GDP.MKTP.KD.ZG': 'GDP_Growth',
        'KOR_SL.UEM.TOTL.ZS': 'Unemployment',
        'KOR_NE.CON.PRVT.CD': 'Private_Consumption',
        'KRWJPY': 'FX_vs_JPY'
    }, inplace=True)
    df_korea_econ['country'] = 'South Korea'

    # 中国と韓国の経済データを結合
    df_econ_processed = pd.concat([df_china_econ, df_korea_econ], ignore_index=True)
    print("\n--- Processed Economic Data (head) ---")
    print(df_econ_processed.head())
    print("\n--- Processed Economic Data (info) ---")
    df_econ_processed.info()


    # 訪問者数データ df_yearly_country と結合
    # df_yearly_country の国名を英語にマッピング (既に行っている場合は、その結果のDataFrameを使用)
    # country_name_mapping は既に定義されている想定
    # df_yearly_country['country'] は日本語のままなので、英語名に変換する
    if 'country_en' not in df_yearly_country.columns:
         df_yearly_country.loc[:, 'country_en'] = df_yearly_country['country'].map(country_name_mapping)

    # 経済データと訪問者数データを結合
    # 経済データの'country'列と、訪問者データの'country_en'列でマージ
    # df_yearly_country には 'year', 'country' (日本語), 'visitors', 'country_en' (英語), 'visitors_growth_rate' がある
    df_merged_econ = pd.merge(
        df_yearly_country[df_yearly_country['country_en'].isin(['China', 'South Korea'])], # 韓国と中国の訪問者データのみ
        df_econ_processed,
        left_on=['year', 'country_en'],
        right_on=['year', 'country'],
        how='left' # 訪問者データに経済データを紐づける
    )
    # 重複する可能性のある 'country_y' を削除 (もしあれば)
    if 'country_y' in df_merged_econ.columns:
        df_merged_econ.drop(columns=['country_y'], inplace=True)
    if 'country_x' in df_merged_econ.columns: # 'country_x' を 'country' にリネーム (元々の日本語国名)
        df_merged_econ.rename(columns={'country_x': 'country_jp'}, inplace=True)


    print("\n--- Merged Visitor and Economic Data (head) ---")
    print(df_merged_econ.head())
    print("\n--- Merged Visitor and Economic Data (info) ---")
    df_merged_econ.info()

    # この df_merged_econ を使って、次のステップで可視化や相関分析を行う
    # 例えば、2019年と2024年のGDP成長率や為替レートを比較表示するなど。

    # 例: 2019年と2024年の韓国と中国の主要指標を表示
    comparison_years_econ = [2019, 2024]
    df_comparison_econ = df_merged_econ[df_merged_econ['year'].isin(comparison_years_econ) & df_merged_econ['country_en'].isin(['South Korea', 'China'])].copy()
    
    print("\n--- Economic Comparison: South Korea vs China (2019 & 2024) ---")
    # 必要な列を選択して表示
    print(df_comparison_econ[['year', 'country_en', 'visitors', 'visitors_growth_rate', 'GDP_Growth', 'Unemployment', 'FX_vs_JPY']])

# %%

# %% [markdown]
# ### 10. 訪問者数と経済指標・為替レートの関連性可視化 (韓国・中国)
# - 統合データ `df_merged_econ` を使用して、訪問者数と主要な経済指標（GDP成長率、為替レートなど）の関連性をグラフで可視化します。

# %%
if 'df_merged_econ' in globals() and df_merged_econ is not None and not df_merged_econ.empty:
    print("\n--- Visualizing Visitor Numbers vs. Economic Indicators ---")

    # 可視化対象の指標リスト
    economic_indicators = ['GDP_Growth', 'Unemployment', 'Private_Consumption', 'FX_vs_JPY']
    # 念のため、これらの列を数値型に変換 (エラーはNaNに)
    for col in economic_indicators + ['visitors']:
        if col in df_merged_econ.columns:
            df_merged_econ[col] = pd.to_numeric(df_merged_econ[col], errors='coerce')

    # 国ごとにグラフを作成
    for country_name in ['South Korea', 'China']:
        df_country_econ = df_merged_econ[df_merged_econ['country_en'] == country_name].copy()
        df_country_econ.sort_values(by='year', inplace=True) # 年でソート

        if df_country_econ.empty:
            print(f"No data for {country_name} to visualize.")
            continue

        # --- グラフ1: 訪問者数と主要経済指標の推移 --- 
        # GDP成長率と為替レートを対象とする
        indicators_for_trend = ['GDP_Growth', 'FX_vs_JPY']

        for indicator in indicators_for_trend:
            if indicator not in df_country_econ.columns or df_country_econ[indicator].isnull().all():
                print(f"Skipping trend plot for {country_name} - {indicator} due to missing data.")
                continue

            fig, ax1 = plt.subplots(figsize=(12, 6))

            # 左軸: 訪問者数
            color = 'tab:blue'
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Number of Visitors (Millions)', color=color)
            ax1.plot(df_country_econ['year'], df_country_econ['visitors'] / 1_000_000, color=color, marker='o', label='Visitors (M)')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1fM'))

            # 右軸: 経済指標
            ax2 = ax1.twinx() # 共通のX軸を持つ第二のY軸
            color = 'tab:red'
            ax2.set_ylabel(indicator, color=color) # 経済指標のラベル
            ax2.plot(df_country_econ['year'], df_country_econ[indicator], color=color, marker='s', linestyle='--', label=indicator)
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title(f'Visitor Trends vs. {indicator} - {country_name}')
            fig.tight_layout() # レイアウト調整
            # 凡例の表示 (ax1とax2の凡例をまとめる)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left')
            plt.xticks(df_country_econ['year'].unique()) # 各年のラベルを表示
            plt.grid(True)
            plt.show()

        # --- グラフ2: 訪問者数 vs. 各経済指標の散布図 --- 
        for indicator in economic_indicators:
            if indicator not in df_country_econ.columns or df_country_econ[indicator].isnull().all() or df_country_econ['visitors'].isnull().all():
                print(f"Skipping scatter plot for {country_name} - {indicator} vs Visitors due to missing data.")
                continue
            
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df_country_econ, x=indicator, y='visitors')
            # 相関係数を計算してタイトルに表示 (NaNを無視)
            correlation = df_country_econ[indicator].corr(df_country_econ['visitors'])
            plt.title(f'Visitors vs. {indicator} - {country_name}\nCorrelation: {correlation:.2f}')
            plt.xlabel(indicator)
            plt.ylabel('Number of Visitors')
            plt.grid(True)
            
            # Y軸をMillion単位に
            ax_scatter = plt.gca()
            ax_scatter.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x/1_000_000:.1f}M'))
            
            plt.tight_layout()
            plt.show()
else:
    print("df_merged_econ is not defined or empty. Skipping visualization.")

# %%

# %% [markdown]
# ### 11. パンデミック期間を除いた経済指標との関連再分析 (韓国・中国)
# - 2020年～2022年のデータを除外し、Private_Consumption および FX_vs_JPY と訪問者数の関連を再度確認します。

# %%
if 'df_merged_econ' in globals() and df_merged_econ is not None and not df_merged_econ.empty:
    print("\n--- Re-analyzing Visitor Numbers vs. Economic Indicators (excluding 2020-2022) ---")

    # パンデミック期間を除外したデータフレームを作成
    years_to_exclude_pandemic = [2020, 2021, 2022]
    df_merged_econ_filtered = df_merged_econ[~df_merged_econ['year'].isin(years_to_exclude_pandemic)].copy()

    if df_merged_econ_filtered.empty:
        print("No data available after excluding pandemic years.")
    else:
        # 分析対象の指標
        indicators_for_reanalysis = ['Private_Consumption', 'FX_vs_JPY']

        for country_name in ['South Korea', 'China']:
            df_country_filtered = df_merged_econ_filtered[df_merged_econ_filtered['country_en'] == country_name].copy()

            if df_country_filtered.empty:
                print(f"No data for {country_name} after excluding pandemic years.")
                continue
            
            print(f"\n--- Scatter Plots for {country_name} (excluding 2020-2022) ---")
            for indicator in indicators_for_reanalysis:
                # 指標の列が存在し、NaNでない値があるか確認
                if indicator not in df_country_filtered.columns or df_country_filtered[indicator].isnull().all() or df_country_filtered['visitors'].isnull().all():
                    print(f"Skipping scatter plot for {country_name} - {indicator} vs Visitors due to missing or all NaN data.")
                    continue

                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=df_country_filtered, x=indicator, y='visitors')
                
                # 相関係数を計算してタイトルに表示 (NaNを無視)
                # 欠損値を除外して相関を計算するために dropna() を使用
                correlation_filtered = df_country_filtered[[indicator, 'visitors']].dropna().corr().iloc[0, 1]
                
                plt.title(f'Visitors vs. {indicator} - {country_name} (excl. 2020-22)\nCorrelation: {correlation_filtered:.2f}')
                plt.xlabel(indicator)
                plt.ylabel('Number of Visitors')
                plt.grid(True)
                
                # Y軸をMillion単位に
                ax_scatter_filtered = plt.gca()
                ax_scatter_filtered.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x/1_000_000:.1f}M'))
                
                plt.tight_layout()
                plt.show()
else:
    print("df_merged_econ is not defined or empty. Skipping re-analysis.")

# %%

# %% [markdown]
# ### 12. 2019年 vs 2024年: Private Consumption と訪問者数の比較 (韓国・中国)
# - 韓国と中国について、2019年と2024年の Private Consumption の実数値と訪問者数、およびそれぞれの変化率を比較します。

# %% [markdown]
# **参考: 中国・韓国の2024年の個人消費支出に関する補足データ (提供情報より)**
# - 中国: 一人当たり消費支出 20,631元 (前年比 +5.6%), サービス消費支出 (一人当たり) +7.4%
# - 韓国: 名目個人最終消費支出 (合計値) 1,235,320,416百万ウォン (前年比 +3.2%)
# - ※これらのデータは、既存データの Private_Consumption (不変実質値の合計値) と定義・単位が異なるため、直接結合せず参考情報として活用します。

# %%
if 'df_merged_econ' in globals() and df_merged_econ is not None and not df_merged_econ.empty:
    print("\n--- Comparing Private Consumption and Visitor Numbers (2019 vs 2024) ---")
    
    comparison_pc_data = []
    years_for_pc_comp = [2019, 2024]

    for country_name in ['South Korea', 'China']:
        df_country_comp = df_merged_econ[ 
            (df_merged_econ['country_en'] == country_name) & 
            (df_merged_econ['year'].isin(years_for_pc_comp))
        ].copy()
        
        if len(df_country_comp) < 2: # 2019年と2024年の両方のデータがない場合はスキップ
            print(f"Skipping Private Consumption comparison for {country_name} due to missing data for 2019 or 2024.")
            continue
        
        # 念のため年でソートして、古い年(2019)を最初に持ってくる
        df_country_comp.sort_values(by='year', inplace=True)

        try:
            visits_2019 = df_country_comp[df_country_comp['year'] == 2019]['visitors'].iloc[0]
            pc_2019 = df_country_comp[df_country_comp['year'] == 2019]['Private_Consumption'].iloc[0]
            
            visits_2024 = df_country_comp[df_country_comp['year'] == 2024]['visitors'].iloc[0]
            pc_2024 = df_country_comp[df_country_comp['year'] == 2024]['Private_Consumption'].iloc[0]

            # 変化率を計算 (0除算を避ける)
            visitor_change_pct = ((visits_2024 - visits_2019) / visits_2019) * 100 if visits_2019 else float('nan')
            pc_change_pct = ((pc_2024 - pc_2019) / pc_2019) * 100 if pc_2019 else float('nan')

            comparison_pc_data.append({
                'Country': country_name,
                'Year': 2019,
                'Visitors': visits_2019,
                'Private_Consumption': pc_2019
            })
            comparison_pc_data.append({
                'Country': country_name,
                'Year': 2024,
                'Visitors': visits_2024,
                'Private_Consumption': pc_2024
            })
            comparison_pc_data.append({
                'Country': country_name,
                'Year': '2019-2024 Change (%)',
                'Visitors': visitor_change_pct,
                'Private_Consumption': pc_change_pct
            })
        except IndexError:
            print(f"Could not retrieve data for {country_name} for 2019 or 2024 for PC comparison.")
            continue
        except ZeroDivisionError:
            print(f"Zero division error during change rate calculation for {country_name}.")
            continue # 0除算エラーが発生した場合も次の国へ
            
    if comparison_pc_data:
        df_pc_comparison_summary = pd.DataFrame(comparison_pc_data)
        # Pandasの表示設定を変更して、科学的記数法を使わないようにする
        pd.options.display.float_format = '{:.2f}'.format # 小数点以下2桁で表示
        print("\n--- Private Consumption and Visitor Numbers Comparison (2019 vs 2024) ---")
        print(df_pc_comparison_summary.set_index(['Country', 'Year']))
        # 表示設定をデフォルトに戻す (他の部分に影響しないように)
        pd.reset_option('display.float_format')
    else:
        print("No data to display for Private Consumption comparison.")
else:
    print("df_merged_econ is not defined or empty. Skipping Private Consumption comparison.")

# %%

# %% [markdown]
# ### 13. Private Consumption の推移確認 (2019 vs 2023)
# - Private Consumption の2024年のデータが欠損しているため、データのある直近の年である2023年までの推移を2019年と比較し、傾向を確認します。提供された2024年の前年比データと合わせて考察します。

# %%
if 'df_merged_econ' in globals() and df_merged_econ is not None and not df_merged_econ.empty:
    print("\n--- Private Consumption Trends (2019 vs 2023) ---")
    
    pc_trend_data = []
    years_for_pc_trend = [2019, 2023]

    for country_name in ['South Korea', 'China']:
        df_country_trend = df_merged_econ[ 
            (df_merged_econ['country_en'] == country_name) & 
            (df_merged_econ['year'].isin(years_for_pc_trend))
        ].copy()
        
        if len(df_country_trend) < 2: # 2019年と2023年の両方のデータがない場合はスキップ
            print(f"Skipping Private Consumption trend for {country_name} due to missing data for 2019 or 2023.")
            continue
        
        # 年でソート
        df_country_trend.sort_values(by='year', inplace=True)

        try:
            pc_2019 = df_country_trend[df_country_trend['year'] == 2019]['Private_Consumption'].iloc[0]
            pc_2023 = df_country_trend[df_country_trend['year'] == 2023]['Private_Consumption'].iloc[0]
            
            # 変化率を計算 (0除算を避ける)
            pc_change_pct_2019_2023 = ((pc_2023 - pc_2019) / pc_2019) * 100 if pc_2019 and not pd.isna(pc_2019) else float('nan')

            pc_trend_data.append({
                'Country': country_name,
                'Year': 2019,
                'Private_Consumption (Constant LCU)': pc_2019
            })
            pc_trend_data.append({
                'Country': country_name,
                'Year': 2023,
                'Private_Consumption (Constant LCU)': pc_2023
            })
            pc_trend_data.append({
                'Country': country_name,
                'Year': '2019-2023 Change (%)',
                'Private_Consumption (Constant LCU)': pc_change_pct_2019_2023
            })

        except IndexError:
             print(f"Could not retrieve data for {country_name} for 2019 or 2023 for PC trend.")
             continue
        except ZeroDivisionError:
            print(f"Zero division error during change rate calculation (2019-2023) for {country_name}.")
            continue

    if pc_trend_data:
        df_pc_trend_summary = pd.DataFrame(pc_trend_data)
        # 表示設定を一時的に変更
        pd.options.display.float_format = '{:.2f}'.format # 小数点以下2桁で表示
        print("\n--- Private Consumption Comparison (2019 vs 2023) ---")
        print(df_pc_trend_summary.set_index(['Country', 'Year']))
        # 表示設定をデフォルトに戻す
        pd.reset_option('display.float_format')

    else:
        print("No data to display for Private Consumption trend.")

else:
    print("df_merged_econ is not defined or empty. Skipping Private Consumption trend analysis.")

# %%
