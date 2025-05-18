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
