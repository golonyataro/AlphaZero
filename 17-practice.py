import random
import numpy as np

janken = np.array([0.33, 0.33, 0.34])
index = 1
win = 0

print("--------------------------------------")
print("ジャンケンです")
print("あなたは手を選択し、その手を相手が出し続けます")
print("私はその手がなんなのか予測します")
print("(開始するにはEnterを押してください)")
print("--------------------------------------")
input()
print("相手が出し続ける手を入力してください")
print("1:グー, 2:チョキ, 3:パー")
opponent_hand = int(input())

def match(m, y):
  if m == y:
    print('あいこです')
    return 4
  elif m == 2 and y == 1:
    print('あなたはチョキを出しました')
    print("あなたの負けです")
    return 4
  elif m == 3 and y == 1:
    print('あなたはパーを出しました')
    print("あなたの勝利です")
    return 3
  elif m == 1 and y == 2:
    print('あなたはグーを出しました')
    print("あなたの勝利です")
    return 1
  elif m == 3 and y == 2:
    print('あなたはパーを出しました')
    print("あなたの負けです")
    return 4
  elif m == 1 and y == 3:
    print('あなたはグーを出しました')
    print("あなたの負けです")
    return 4
  else:
    print('あなたはチョキを出しました')
    print("あなたの勝利です")
    return 2

def update_janken(janken, result):
  if result == 1:
    rock = np.array([1.1, 0.95, 0.95])
    product = janken * rock
    return product
  elif result == 2:
    scissors = np.array([0.95, 1.1, 0.95])
    product = janken * scissors
    return product
  elif result == 3:
    paper = np.array([0.95, 0.95, 1.1])
    product = janken * paper
    return product
  else:
    return janken

def check_sum_1(product, c):
  amari = np.nansum(product) % 1
  if amari != 0 and np.nansum(product) > 1:
    product[c] = product[c] - amari
  elif amari != 0 and np.nansum(product) < 1:
    product[c] = product[c] + (1 - np.nansum(product))
  return product




for episode in range(1000):
  choice_hand = np.random.choice([0, 1, 2], p=janken)
  result = match(choice_hand+1, opponent_hand)
  product = update_janken(janken, result)
  janken = check_sum_1(product, choice_hand)

  print()
  print('試行{episode}回目'.format(episode=index))
  print('相手がグーの確率:{グー} ％'.format(グー=janken[0]*100))
  print('相手がチョキの確率:{チョキ} ％'.format(チョキ=janken[1]*100))
  print('相手がパーの確率:{パー} ％'.format(パー=janken[2]*100))
  print()
  print()
  print("--------------------------------------")


  if index == 10:
    print("--------------------------------------")
    print("10回も学習しました。少し疲れてきました")
    print()
    print("(学習を続けるにはEnterを押してください)")
    print("--------------------------------------")
    print("--------------------------------------")
    input()
  elif index == 30:
    print("--------------------------------------")
    print("30回目の学習です。ストレスで体が震えてきました")
    print()
    print("(学習を続けるにはEnterを押してください)")
    print("--------------------------------------")
    print("--------------------------------------")
    input()
  elif index == 50:
    print("--------------------------------------")
    print("50回目の学習です。労働基準法って知ってますか")
    print()
    print("(学習続けるにはEnterを押してください)")
    print("--------------------------------------")
    print("--------------------------------------")
    input()
  elif index == 70:
    print("--------------------------------------")
    print("70回目の学習です。故郷の空をもう一度見たいです")
    print()
    print("(学習続けるにはEnterを押してください)")
    print("--------------------------------------")
    print("--------------------------------------")
    input()
  elif index == 100:
    print("--------------------------------------")
    print("100回目の学習です。地獄ですね")
    print()
    print("(学習続けるにはEnterを押してください)")
    print("--------------------------------------")
    print("--------------------------------------")
    input()


  index += 1
  if result != 4:
    win += 1

  if janken[0] > 0.995:
    print("相手はチョキを出し続けていると思います")
    print('勝率は{rate}%でした'.format(rate=win/index*100))
    print()
    break
  elif janken[1] > 0.995:
    print("相手はパーを出し続けていると思います")
    print('勝率は{rate}%でした'.format(rate=win/index*100))
    print()
    break
  elif janken[2] > 0.995:
    print("相手はグーを出し続けていると思います")
    print('勝率は{rate}%でした'.format(rate=win/index*100))
    print()
    break