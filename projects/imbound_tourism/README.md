# 訪日観光客 探索的データ分析

このプロジェクトは、訪日観光客データの探索的データ分析をPythonと一般的なデータサイエンスライブラリを用いて行います。

## セットアップ

プロジェクトの依存関係を管理するために、仮想環境を使用することを強く推奨します。

1.  **プロジェクトディレクトリへ移動:**
    ```bash
    cd projects/imbound_tourism/
    ```

2.  **仮想環境を作成:**
    Python 3がインストールされていることを確認してください。
    ```bash
    python3 -m venv venv
    ```

3.  **仮想環境をアクティベート:**

    *   **macOS / Linux:**
        ```bash
        source venv/bin/activate
        ```
    *   **Windows:**
        ```cmd
        venv\Scripts\activate
        ```
    ターミナルプロンプトの先頭に `(venv)` と表示されていれば、仮想環境が有効になっています。

4.  **依存ライブラリをインストール:**
    `requirements.txt` ファイルに記載された必要なライブラリをインストールします。
    ```bash
    pip install -r requirements.txt
    ```

## スクリプトの実行

仮想環境をアクティベートし、依存ライブラリをインストールした後、分析スクリプトを実行できます。

```bash
# プロジェクトディレクトリ (projects/imbound_tourism/) にいることを確認してください
python scripts/exploratory_analysis.py
```

## 仮想環境をディアクティベート

プロジェクトでの作業が完了したら、仮想環境を無効にできます。

```bash
deactivate
``` 