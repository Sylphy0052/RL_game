# 強化学習について

## 2020年

* 価値ベース，離散制御アルゴリズム
  * NGU
  * Agent57
* 方策ベース，連続値制御アルゴリズム
  * SimPLe

## 2016~2019年

* 価値ベース，離散制御アルゴリズム
  * Rainbow
  * R2D2
* 方策ベース，連続値制御アルゴリズム
  * UNREAL

## 流れ

* Q学習
* DQN
* Rainbow
* Ape-X
* R2D2
* Agent57

## Chap01

特になし

## Chap02

Q学習

TD学習: TD誤差を計算して状態価値を更新する(`r_t + gamma * V(st+1) - V(st)`)

更新式: `V(st)←V(st)+a(rt+gamma * V(st+1)-V(st))`

報酬 + 学習率 * 次の状態価値 - 状態価値

Q値更新式: `Q(st,at)←Q(st,at)+a(rt+gamma * maxp Q(st+1,p)-Q(st,at))`

* V: 状態価値
* Q: 行動価値

## 参考

* [chap01](https://qiita.com/pocokhc/items/a8120b0abd5941dd7a9f)
* [chap02](https://qiita.com/pocokhc/items/8ed40be84a144b28180d)
