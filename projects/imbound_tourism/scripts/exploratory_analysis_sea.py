# %% [markdown]
# # タイ・ラオス・カンボジアからの訪問者数分析

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import seaborn as sns

# フォント関連の初期設定 (既存スクリプトと同様)
try:
    fm._rebuild()
except AttributeError:
    pass # 古いバージョンのmatplotlibなど
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
plt.rcdefaults() # matplotlibのデフォルト設定を使用

# %% [markdown]
# ### 1. データの読み込みと前処理

# %%
df_raw = pd.read_csv('../data/visitors.csv')
print("--- df_raw.head() ---")
print(df_raw.head())

# 注釈行の削除
df_cleaned = df_raw[~df_raw['country'].str.startswith('注１', na=False) & ~df_raw['country'].str.startswith('注２', na=False)].copy()

# visitors列を数値型に変換
df_cleaned.loc[:, 'visitors'] = pd.to_numeric(df_cleaned['visitors'], errors='coerce')

# 国名がNaNまたは空欄の行を除外
df_cleaned.dropna(subset=['country'], inplace=True)
df_cleaned = df_cleaned[df_cleaned['country'].str.strip() != '']

# 集計行を除外 (例: '総数', '計' で終わるもの)
# 「その他」も除外対象に含めるかは、これらの国々が含まれうるか考慮が必要ですが、一旦既存のロジックを踏襲します。
excluded_keywords = ['総数', '計', 'その他']
pattern = '|'.join(excluded_keywords)
df_cleaned = df_cleaned[~df_cleaned['country'].str.contains(pattern, na=False)]

# 月を数値に変換する関数
def convert_month_to_int(month_str):
    if isinstance(month_str, str):
        return int(month_str.replace('月', ''))
    return np.nan

df_cleaned.loc[:, 'month_num'] = df_cleaned['month'].apply(convert_month_to_int)

# date列を作成
df_cleaned.dropna(subset=['year', 'month_num'], inplace=True)
df_cleaned.loc[:, 'date'] = pd.to_datetime(df_cleaned['year'].astype(int).astype(str) + '-' +
                                     df_cleaned['month_num'].astype(int).astype(str) + '-' + '1', errors='coerce')

# 不要列の削除とNaNデータの除去
df_cleaned.drop(columns=['month_num'], inplace=True)
df_cleaned.dropna(subset=['visitors', 'date'], inplace=True)
df_cleaned.loc[:, 'visitors'] = df_cleaned['visitors'].astype(int)

# 2025年のデータを除外
df_cleaned = df_cleaned[df_cleaned['year'] != 2025].copy()

print("\\n--- df_cleaned.info() after basic cleaning ---")
df_cleaned.info()
print("\\n--- df_cleaned.head() after basic cleaning ---")
print(df_cleaned.head())

# print("\\n--- Unique country names in df_cleaned ---") # デバッグ用なのでコメントアウト
# print(df_cleaned['country'].unique()) # デバッグ用なのでコメントアウト


# %% [markdown]
# ### 2. 対象国（タイ）のデータ抽出

# %%
# 対象国のリスト (日本語)
target_country_jp = ['タイ'] # タイに限定

# 対象国のみのデータを抽出
df_thailand = df_cleaned[df_cleaned['country'].isin(target_country_jp)].copy()

if df_thailand.empty:
    print("\\nError: No data found for Thailand. Please check the country name 'タイ' in visitors.csv.")
else:
    print(f"\\n--- Data extracted for タイ ---")
    print(f"Number of rows: {len(df_thailand)}")
    print(df_thailand.head())

    # 国名マッピング (英語へ)
    # タイ専用なので、country_en列は必須ではないが、他スクリプトとの整合性や将来性を考え付与
    df_thailand.loc[:, 'country_en'] = 'Thailand' 

    print("\\n--- df_thailand.head() with English country name ---")
    print(df_thailand.head())
    # print("\\n--- df_thailand['country_en'].value_counts() ---") # デバッグ用なのでコメントアウト
    # print(df_thailand['country_en'].value_counts()) # デバッグ用なのでコメントアウト
    # print("\\n--- df_thailand.info() after adding country_en ---") # デバッグ用なのでコメントアウト
    # df_thailand.info() # デバッグ用なのでコメントアウト


# %% [markdown]
# ### 3. 年間訪問者数の集計 (タイ)

# %%
if not df_thailand.empty:
    # タイの年間訪問者数
    df_yearly_thailand = df_thailand.groupby('year')['visitors'].sum().reset_index()
    print("\\n--- Annual Visitors from Thailand ---")
    print(df_yearly_thailand.head())
    # df_yearly_country_sea の代わりに df_yearly_thailand を使用するため、以下の行は不要または修正対象
    # print("\\n--- df_yearly_country_sea['country_en'].value_counts() ---") # デバッグ用、かつ複数国前提なので削除
    # print(df_yearly_country_sea['country_en'].value_counts()) # デバッグ用、かつ複数国前提なので削除

    # 三国合計の年間訪問者数の処理は不要になる
    # df_yearly_total_sea_sum = df_sea.groupby('year')['visitors'].sum().reset_index()
    # df_yearly_total_sea_sum.rename(columns={'visitors': 'total_visitors_sea'}, inplace=True)
    # print("\\n--- Total Annual Visitors (Thailand, Laos, Cambodia combined) ---")
    # print(df_yearly_total_sea_sum.head())
else:
    print("\\nSkipping annual aggregation as no data was found for Thailand.")

# %%
# %% [markdown]
# ### 4. 年間訪問者数の推移グラフ (タイ)

# %%
# if 'df_yearly_country_sea' in globals() and not df_yearly_country_sea.empty: # このブロックはタイ単独の場合不要になるのでコメントアウトまたは削除
#     plt.figure(figsize=(12, 6))
#     sns.lineplot(data=df_yearly_country_sea, x='year', y='visitors', hue='country_en', marker='o')
#     plt.title('Annual Visitor Trends (Thailand, Laos, Cambodia)')
#     plt.xlabel('Year')
#     plt.ylabel('Number of Visitors')
#     plt.grid(True)
#     plt.xticks(df_yearly_country_sea['year'].unique()) # 各年のラベルを表示
#     plt.legend(title='Country')
#     
#     ax = plt.gca()
#     def custom_formatter(x, pos):
#         if x >= 1_000_000:
#             return f'{x/1_000_000:.1f}M'
#         elif x >= 1_000:
#             return f'{x/1_000:.0f}K'
#         else:
#             return f'{x:.0f}'
#     ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter))
#     
#     plt.tight_layout()
#     plt.show()
# else:
#     print("\\nSkipping visitor trends plot as df_yearly_country_sea is not available.")

if 'df_yearly_thailand' in globals() and not df_yearly_thailand.empty:
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_yearly_thailand, x='year', y='visitors', marker='o', color='green') # 色を任意で変更
    plt.title('Annual Visitor Trends from Thailand')
    plt.xlabel('Year')
    plt.ylabel('Number of Visitors')
    plt.grid(True)
    plt.xticks(df_yearly_thailand['year'].unique()) # 各年のラベルを表示
    
    ax = plt.gca()
    # Y軸を適切な単位にフォーマット (例: K単位、M単位)
    def custom_formatter(x, pos):
        if x >= 1_000_000:
            return f'{x/1_000_000:.1f}M'
        elif x >= 1_000:
            return f'{x/1_000:.0f}K'
        else:
            return f'{x:.0f}'
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter))
    
    plt.tight_layout()
    plt.show()
else:
    print("\\nSkipping Thailand visitor trends plot as df_yearly_thailand is not available.")


# %%
# %% [markdown]
# ### 5. 年間訪問者数 (棒グラフ - タイ)

# %%
if 'df_yearly_thailand' in globals() and not df_yearly_thailand.empty:
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_yearly_thailand, x='year', y='visitors', color='skyblue') # hueは不要
    
    plt.title('Annual Visitors from Thailand (Bar Chart)')
    plt.xlabel('Year')
    plt.ylabel('Number of Visitors')
    plt.xticks(rotation=45)
    # plt.legend(title='Country') # 単独国なので凡例は不要
    plt.grid(axis='y') # Y軸のみグリッド表示
    
    # Y軸のフォーマット (推移グラフで定義したものを再利用)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter))
    
    plt.tight_layout()
    plt.show()
else:
    print("\\nSkipping bar chart as df_yearly_thailand is not available.")

# %%
# %% [markdown]
# ### 6. 年間訪問者数の前年比成長率 (タイ)

# %%
if 'df_yearly_thailand' in globals() and not df_yearly_thailand.empty:
    # 前年比成長率を計算
    # df_yearly_country_seaではなくdf_yearly_thailandから計算
    # country_enでグループ化する必要がなくなる
    df_yearly_thailand['growth_rate_yoy'] = df_yearly_thailand['visitors'].pct_change() * 100
    df_growth_yoy_thailand = df_yearly_thailand.dropna(subset=['growth_rate_yoy']).copy() # NaNを除外 (初年度など)

    if not df_growth_yoy_thailand.empty:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_growth_yoy_thailand, x='year', y='growth_rate_yoy', marker='o', color='orange') # hue削除、色指定
        plt.title('Year-over-Year Visitor Growth Rate from Thailand')
        plt.xlabel('Year')
        plt.ylabel('Growth Rate (%)')
        plt.grid(True)
        plt.xticks(df_growth_yoy_thailand['year'].unique()) # 各年のラベルを表示
        # plt.legend(title='Country') # 単独国なので凡例は不要
        plt.axhline(0, color='grey', lw=1, linestyle='--') # 0%のラインを追加
        plt.tight_layout()
        plt.show()
    else:
        print("\\nNo data to plot for YoY growth rate after filtering.")
else:
    print("\\nSkipping YoY growth rate plot as df_yearly_thailand is not available.")

# %%
# %% [markdown]
# ### 7. 年間訪問者数の2019年比成長率 (タイ)

# %%
if 'df_yearly_thailand' in globals() and not df_yearly_thailand.empty:
    # 2019年の訪問者数を取得
    # 国別にする必要がないため、単純に2019年の値を取得
    try:
        visitors_2019_th = df_yearly_thailand[df_yearly_thailand['year'] == 2019]['visitors'].iloc[0]
    except IndexError:
        visitors_2019_th = np.nan # 2019年のデータがない場合はNaN
        print("\\nWarning: 2019 data not found for Thailand. Skipping 2019-based growth rate calculation.")

    if not pd.isna(visitors_2019_th) and visitors_2019_th != 0:
        df_growth_vs_2019_thailand = df_yearly_thailand.copy()
        df_growth_vs_2019_thailand['growth_rate_vs_2019'] = \
            ((df_growth_vs_2019_thailand['visitors'] - visitors_2019_th) / visitors_2019_th) * 100
        
        # 2019年以降のデータをプロット
        df_plot_growth_vs_2019_thailand = df_growth_vs_2019_thailand[df_growth_vs_2019_thailand['year'] >= 2019].copy()

        if not df_plot_growth_vs_2019_thailand.empty:
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=df_plot_growth_vs_2019_thailand, x='year', y='growth_rate_vs_2019', marker='o', color='teal') # hue削除、色指定
            plt.title('Visitor Growth Rate vs. 2019 from Thailand')
            plt.xlabel('Year')
            plt.ylabel('Growth Rate vs. 2019 (%)')
            plt.grid(True)
            unique_years_vs_2019 = sorted(df_plot_growth_vs_2019_thailand['year'].unique())
            plt.xticks(unique_years_vs_2019)
            # plt.legend(title='Country') # 単独国なので凡例は不要
            plt.axhline(0, color='grey', lw=1, linestyle='--') # 0%のライン (2019年の水準)
            plt.tight_layout()
            plt.show()
        else:
            print("\\nNo data to plot for growth rate vs 2019 after filtering.")
    elif pd.isna(visitors_2019_th):
        pass # 警告は上で表示済み
    else: # visitors_2019_thが0の場合
        print("\\nCannot calculate growth rate vs 2019 because 2019 visitor count is zero.")
else:
    print("\\nSkipping 2019-based growth rate plot as df_yearly_thailand is not available.")

# %% 