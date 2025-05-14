# データ分析環境 (Docker + Jupyter Lab)

## 起動手順

1.  **Dockerイメージのビルド:**
    ```bash
    docker build -t my-jupyter-lab .
    ```

2.  **Dockerコンテナの実行:**
    ローカルのワークスペース全体をコンテナの `/app` ディレクトリにマウントします。
    ```bash
    docker run -d -p 8888:8888 -v $(pwd):/app --name my-jupyter-container my-jupyter-lab
    ```

3.  **Jupyter Labへのアクセス:**
    ブラウザで `http://localhost:8888` を開きます。

## データ永続化について
ローカルのワークスペース全体がコンテナ内の `/app` にマウントされているため、コンテナを停止・削除しても `/app` 以下のプロジェクトファイルやデータはローカルに保持されます。
