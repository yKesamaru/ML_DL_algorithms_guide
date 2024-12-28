## はじめに
わたしは機械学習の色々な媒体を見ていて「これってなんだっけ？」がよく生じます。最近だと[競馬AI などのギャンブル AI を作る前に考えること
](https://zenn.dev/junichiro/articles/b15a9964a4507d)という素晴らしい記事を拝見していた時に「LGBMってなんだっけ？」と思いました。そうやって考えているとXGBoostとかcatBoostとか伝言ゲームのように浮かんできて「あ〜網羅的に見直したいな」という欲求が生じたのですがネットですぐに見つかりませんでした。（あった: [機械学習の分類](https://qiita.com/kenjihiranabe/items/992cf18e03e7f884f7e7)）

手元に「あたらしい人工知能の教科書」という素晴らしい本があるのですが、初版が古いのと「式まではいらん」という感じで。
なので自分用に用意しました。

![](assets/eye-catch.png)

## 注意
- 自分用なので「6GB VRAM」で使用できるかどうかなどが書いてあります。
- それとむりくり10分類してあります。SPADEなど重複して登場します。
- 細部に関して一応目を通してありますが「UNet 医療画像セグメンテーションに特化したネットワーク。 (6GB vRAM: △)」とか、いや別に医療用じゃないでしょ的なところがありますのでお気をつけください。
- なにかの本などでざっくり全体像を読んだことがある方用です。

## 使い方
- 眺める（たのしい）

## 6GB VRAM
- **6GB vRAM: ○** … 一般的なサイズ・条件であれば問題なく利用可能
- **6GB vRAM: △** … モデルサイズ・バッチサイズ・入力サイズ等の工夫次第で可能（もしくは推論のみ可能）
- **6GB vRAM: ×** … 一般的な学習はまず難しい（推論なら小規模モデルでギリギリ可能な場合あり）

---

## 1. 構造化データの回帰・分類タスク

### XGBoost (eXtreme Gradient Boosting)
高速で正確な勾配ブースティングアルゴリズム。
(6GB VRAM: ○ ※CPUメインでも可)

### CatBoost (Categorical Boosting)
カテゴリカルデータを扱うのに特化したブースティングアルゴリズム。
(6GB VRAM: ○)

### LightGBM (Light Gradient Boosting Machine)
大規模データの高速処理が可能な勾配ブースティング手法。
(6GB VRAM: ○)

### Random Forest
決定木をランダムに構築してアンサンブル学習を行う手法。
(6GB VRAM: ○)

### Support Vector Machines (SVM)
高次元空間で最適な分類境界を求める手法。
(6GB VRAM: ○ ※CPUメインでも可)

### Logistic Regression
分類タスクで用いられる単純な線形モデル。
(6GB VRAM: ○)

### Linear Regression
回帰タスクで用いられる単純な線形モデル。
(6GB VRAM: ○)

### Extra Trees
ランダムフォレストに似たアンサンブル学習手法。
(6GB VRAM: ○)

### Gradient Boosted Decision Trees (GBDT)
勾配ブースティングを用いた決定木のアンサンブル。
(6GB VRAM: ○)

### Stacked Generalization (Stacking)
複数のモデルを組み合わせて予測性能を向上させる手法。
(6GB VRAM: ○)

### AdaBoost (Adaptive Boosting)
弱い分類器を連続して改善するブースティング手法。
(6GB VRAM: ○)

### Histogram-Based Gradient Boosting
バケット化した特徴量を用いて計算を高速化する勾配ブースティング。
（例: LightGBM, scikit-learnのHistGradientBoosting）
(6GB VRAM: ○)

---

## 2. 非構造化データ（画像、音声など）の処理

### Convolutional Neural Networks (CNNs)
画像データの処理に特化したニューラルネットワークの総称。
#### EfficientNet
パラメータ効率を高めたCNNアーキテクチャ。
(6GB vRAM: ○ ※B0など小型モデルなら学習可)
#### ResNet (Residual Networks)
残差接続を利用して深いネットワークを学習可能にしたモデル。
(6GB vRAM: △ ※ResNet-50以上はバッチサイズや入力サイズの工夫が必要)
#### Inception Networks
異なるサイズの畳み込みを組み合わせたアーキテクチャ。
(6GB vRAM: △)
#### DenseNet
各層が全ての前層に接続されたCNN。
(6GB vRAM: △)
#### MobileNet
モバイルデバイス向けに軽量化されたCNN。
(6GB vRAM: ○)
#### ShuffleNet
モバイル向けに高速化されたCNN。
(6GB vRAM: ○)

### Vision Transformers (ViTs)
画像データを処理するためのトランスフォーマーモデル。
(6GB vRAM: △ ※小型モデルなら可)

### UNet
医療画像セグメンテーションに特化したネットワーク。
(6GB vRAM: △)

### Mask R-CNN
インスタンスセグメンテーションを行うモデル（物体検出＋セグメンテーション）。
(6GB vRAM: △〜× ※画像サイズやバッチサイズに依存)

---

## 3. 時系列データおよび依存関係の解析

### Recurrent Neural Networks (RNNs)
時系列データの依存関係を学習するニューラルネットワーク。
#### Long Short-Term Memory Networks (LSTMs)
長期依存を学習可能なRNNの拡張版。
(6GB vRAM: ○)
#### Gated Recurrent Units (GRUs)
LSTMよりもシンプルで高速な時系列モデル。
(6GB vRAM: ○)

### 畳み込みベースの時系列モデル
#### Temporal Convolutional Networks (TCNs)
時間的特徴を捉えるための畳み込みベースのモデル。
(6GB vRAM: ○)
#### WaveNet
音声データの生成や予測に特化したモデル。
(6GB vRAM: △ ※大規模学習には要メモリ)

### Transformer Models
自然言語処理や時系列データに用いられるアーキテクチャ。（時系列用途に拡張したもの）
(6GB vRAM: △〜× ※モデルサイズに依存)

### Prophet
簡単で直感的に使える時系列予測ツール。
(6GB vRAM: ○ ※主にCPU実行)

### ARIMA
時系列データの統計的モデリング手法。
(6GB vRAM: ○)

### Kalman Filters
状態推定に用いられる統計的アルゴリズム。
(6GB vRAM: ○)

---

## 4. 自然言語処理

### Transformer-Based Models
トランスフォーマーベースのモデル群。
#### BERT (Bidirectional Encoder Representations from Transformers)
双方向の文脈を捉える言語モデル。
(6GB vRAM: △〜× ※DistilBERT等の小型版なら学習可)
#### GPT (Generative Pre-trained Transformer)
テキスト生成に特化したモデル。
(6GB vRAM: × ※推論のみ小規模モデルなら可)
#### RoBERTa
BERTを改良した高精度モデル。
(6GB vRAM: △〜×)
#### T5 (Text-to-Text Transfer Transformer)
すべてのタスクをテキスト形式で処理するモデル。
(6GB vRAM: × ※Smallモデルでも工夫次第)
#### ALBERT
軽量化されたBERTモデル。
(6GB vRAM: △ ※Baseクラス程度ならギリギリ)
#### XLNet
双方向と順方向の特性を組み合わせたモデル。
(6GB vRAM: ×)
#### DistilBERT
BERTを簡略化し高速化したモデル。
(6GB vRAM: △)
#### ELECTRA
文の修正検出に基づいた効率的なモデル。
(6GB vRAM: △〜×)

### SpaCy (NLP向けフレームワーク・ツール)
NLPパイプラインの構築に役立つライブラリ／ツール群。
(6GB vRAM: ○ ※GPU要件は内部で使うモデル次第)

---

## 5. データ生成およびモデリング

### Autoencoders
入力データを次元削減して再構築するニューラルネットワーク。
(6GB vRAM: ○)
#### Variational Autoencoders (VAEs)
確率的生成モデルとしてのオートエンコーダー。
(6GB vRAM: △)

### GANs (Generative Adversarial Networks)
生成モデルと識別モデルを競わせて新しいデータを生成。
(6GB vRAM: △〜× ※モデル規模次第)
#### BigGAN など (StyleGAN, CycleGAN, DCGANなど)
高品質な画像生成を実現する代表的なGAN群。
(6GB vRAM: × ※学習には大容量VRAMが必要)

### Flow-based Models
データ分布を直接モデリングする生成モデル。
(6GB vRAM: △〜×)

### Diffusion Models
ノイズを追加・除去して新しいデータを生成。
(6GB vRAM: × ※学習には大容量VRAMが必要)

### PixelCNN
ピクセル単位で画像を生成するモデル。
(6GB vRAM: △)

---

## 6. 非教師あり学習

### Clustering Models
データをグループ化する手法群。
#### K-Means Clustering
データを事前に指定したクラスター数に分割。
(6GB vRAM: ○)
#### DBSCAN
密度ベースでクラスタリング。
(6GB vRAM: ○)
#### Gaussian Mixture Models (GMM)
データ分布を複数のガウス分布で表現。
(6GB vRAM: ○)
#### Hierarchical Clustering
階層的なクラスタリング手法。
(6GB vRAM: ○)
#### Spectral Clustering
グラフ構造を利用したクラスタリング手法。
(6GB vRAM: ○)
#### Agglomerative Clustering
クラスタを統合する階層型手法。
(6GB vRAM: ○)

### Anomaly Detection
正常データから逸脱した異常データを検出する手法群。
#### Isolation Forest
木構造を用いて異常値を識別。
(6GB vRAM: ○)
#### One-Class SVM
正常データの境界を学習して異常値を特定。
(6GB vRAM: ○)
#### SPADE (Spatially Aware Pixel-level Anomaly Detection)
画像における異常ピクセルを検出。
(6GB vRAM: △)
#### PaDiM (Patch Distribution Modeling)
分布をモデル化して異常領域を検出。
(6GB vRAM: △)
#### DeepSVDD (Deep Support Vector Data Description)
異常データの検出に特化したディープラーニングモデル。
(6GB vRAM: △)
#### LOF (Local Outlier Factor)
近傍データとの密度比較で異常を検出する手法。
(6GB vRAM: ○)

---

## 7. 次元削減・特徴抽出

### Principal Component Analysis (PCA)
データの分散を最大化する方向を見つけて次元削減する手法。
(6GB vRAM: ○)
### t-SNE (t-Distributed Stochastic Neighbor Embedding)
高次元データを低次元空間に可視化するための手法。
(6GB vRAM: ○)
### UMAP (Uniform Manifold Approximation and Projection)
t-SNEよりも効率的な次元削減とクラスタリングを可能にする手法。
(6GB vRAM: ○)
### Kernel PCA
非線形データに対応するためにカーネルを使用したPCA。
(6GB vRAM: ○)
### Factor Analysis
観測データの背後にある潜在因子を特定するための手法。
(6GB vRAM: ○)
### Independent Component Analysis (ICA)
相互に独立した特徴を抽出する次元削減手法。
(6GB vRAM: ○)
### Latent Semantic Analysis (LSA)
テキストデータから潜在的な意味を抽出する次元削減手法。
(6GB vRAM: ○)

---

## 8. 強化学習

### Reinforcement Learning (RL)
環境からの報酬に基づいて意思決定を最適化するフレームワーク。
#### Q-Learning
状態-行動ペアの価値を学習するオフポリシー型アルゴリズム。
(6GB vRAM: ○)
#### Deep Q-Networks (DQNs)
Q-Learningをニューラルネットワークで拡張した手法。
(6GB vRAM: △)
#### Policy Gradient Methods (例: REINFORCE)
方策自体を直接最適化する手法。
(6GB vRAM: △)
#### Actor-Critic Methods (例: A3C, PPO)
方策と価値関数を同時に学習する手法。
(6GB vRAM: △)
#### Soft Actor-Critic (SAC)
エンタロピー正則化を用いた安定的なポリシー学習。
(6GB vRAM: △)
#### Deep Deterministic Policy Gradient (DDPG)
連続値アクション空間で使用可能な強化学習手法。
(6GB vRAM: △)
#### Proximal Policy Optimization (PPO)
訓練の安定性と効率を向上させたアルゴリズム。
(6GB vRAM: △)

### Monte Carlo Tree Search (MCTS)
状態空間を探索しながら最適な行動を決定するアルゴリズム。
(6GB vRAM: ○)

---

## 9. グラフ構造データの処理

### Graph Neural Networks (GNNs)
グラフ構造を直接処理するためのニューラルネットワークフレームワーク。
(6GB vRAM: △)
#### GraphSAGE
大規模グラフにおける効率的なノード表現学習。
(6GB vRAM: △)
#### Graph Attention Networks (GATs)
ノード間の重要度を学習するために注意機構を使用した手法。
(6GB vRAM: △)
#### Message Passing Neural Networks (MPNNs)
メッセージ伝播アルゴリズムを基にしたグラフ学習手法。
(6GB vRAM: △)
#### ChebNet (Chebyshev Networks)
スペクトラル畳み込みを効率化したグラフ学習手法。
(6GB vRAM: △)
#### Graph Convolutional Networks (GCNs)
グラフデータ上で畳み込み演算を行う手法。
(6GB vRAM: △)

---

## 10. テキスト分類や異常検知

### Naive Bayes
条件付き確率に基づいてテキストを分類するシンプルなアルゴリズム。
(6GB vRAM: ○)

### Latent Dirichlet Allocation (LDA)
文書をトピック分布に基づいてモデリングする非教師あり手法。
(6GB vRAM: ○)

### Support Vector Machines (SVM, for text classification)
高次元特徴空間で分類境界を最適化するアルゴリズム。
(6GB vRAM: ○)

### Hidden Markov Models (HMMs)
隠れた状態遷移に基づく時系列データやテキスト解析に使用される手法。
(6GB vRAM: ○)

### Anomaly Detection Models
正常データと異常データを区別する手法群。（Isolation Forest等は前述と重複）
#### Isolation Forest
木構造を利用し、異常データを識別。
(6GB vRAM: ○)
#### One-Class SVM
正常データの境界を学習し、異常を検出。
(6GB vRAM: ○)
#### SPADE (Spatially Aware Pixel-level Anomaly Detection)
異常なピクセルを検出する画像解析モデル。
(6GB vRAM: △)
#### PaDiM (Patch Distribution Modeling)
パッチレベルで分布をモデル化し異常を検出。
(6GB vRAM: △)
#### DeepSVDD (Deep Support Vector Data Description)
深層学習を用いた異常検知モデル。
(6GB vRAM: △)
#### LOF (Local Outlier Factor)
近傍データとの密度比較で異常を検出する手法。
(6GB vRAM: ○)

---

## さいごに
辞書眺めたり図鑑見たりするのが好きなんですよね。満足しました。