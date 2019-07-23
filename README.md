# 概要

- ALBERT ガイドラインに従った構成・クラス設計・コード規約・README・CHANGELOG
- API ドキュメントの作成(Sphinx)
- ユニットテスト(動作テスト・検証テスト)を実際に実装  
  参考: https://qiita.com/icoxfog417/items/67764a6756c4548b5fb8
- Linter(flake8)
- Formatter(autopep8)
- 上記 UT, Linter, Formatter と連携する VSCode 設定
- GitHub 用 Issue/PR テンプレート
- direnv による自動仮想環境切り替え・PYTHONPATH 設定

を備えた MNIST の実装例です。

## ディレクトリ構成

<!-- DIRSTRUCTURE_START_MARKER -->
<pre>
mnist/
├─ CHANGELOG ................ 変更履歴
├─ Makefile ................. Makefile
├─ README.md ................ README
├─ requirements.txt ......... Python依存モジュール
├─ docs/ .................... APIドキュメント
├─ models/ .................. 出力モデル・ログ類
├─ notebooks/ ............... 実験用ノートブック
│  ├─ load_digits.ipynb ..... サンプル書き捨てNotebook(1)
│  └─ load_mnist.ipynb ...... サンプル書き捨てNotebook(2)
├─ src/ ..................... ソースコード
│  ├─ loss.py ............... Lossの計算を行うクラス
│  ├─ predictor.py .......... 推論を行い、ラベルを返すクラス
│  └─ models/ ............... モデルクラス
│     ├─ cnn.py ............. CNNモデル実装
│     └─ mlp.py ............. 3層MLPモデル実装
├─ tests/ ................... テストコード
│  ├─ test_mnist_forward.py . 動作テストコード
│  └─ test_mnist_work.py .... 検証テストコード
└─ work/ .................... 実行エントリポイントとなるスクリプトの配置フォルダ
   ├─ predict_mnist.py ...... MNIST推論スクリプト
   └─ train_mnist.py ........ MNIST学習スクリプト
</pre>
<!-- DIRSTRUCTURE_END_MARKER -->

## 環境構築

```bash
$ python3.6 -m venv venv
$ . ./venv/bin/activate
$ pip install -r requirements.txt
```

## 実行方法

### 学習

```bash
$ python work/train_mnist.py
```

#### コマンドラインオプション

| オプション      | 内容                           | デフォルト値 |
| :-------------- | :----------------------------- | :----------- |
| --batchsize, -b | バッチサイズ                   | 100          |
| --epoch, -e     | エポック数                     | 20           |
| --frequency, -f | スナップショット取得間隔       | -1           |
| --out, -o       | モデル出力先                   | models/      |
| --resume, -r    | スナップショットからの学習再開 |              |
| --noplot        | プロットを作成しない           |              |
| --gpu, -g       | 使用 GPU ID(-1:CPU)            | -1           |
| --model, -m     | 使用モデル(mlp/cnn)            | mlp          |
| --unit, -u      | 中間層の数(mlp 使用時のみ)     | 1000         |

### 推論

```bash
$ python work/predict_mnist.py
```

#### コマンドラインオプション

| オプション  | 内容                       | デフォルト値 |
| :---------- | :------------------------- | :----------- |
| --gpu, -g   | 使用 GPU ID(-1:CPU)        | -1           |
| --model, -m | 使用モデル(mlp/cnn)        | mlp          |
| --unit, -u  | 中間層の数(mlp 使用時のみ) | 1000         |

## Makefile

### テスト

```bash
$ make test
```

ユニットテストを実行します。

### API ドキュメント生成

```bash
$ make doc
```

docs/\_build に HTML が出力されます。

### Lint

```bash
$ make lint
```

flake8 を使用して静的解析を行います。

### フォーマット

```bash
$ make format
```

autopep8 を使用してソースコードをフォーマットします。
