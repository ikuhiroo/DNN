#!/usr/bin/env python
#-*- coding: utf-8 -*-

## begin [1-1]
# 必要なモジュールをインポートしよう

from __future__ import print_function

import numpy

import cupy
import chainer
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers

## end [1-1]

## begin [1-3]
# モデルを定義しよう

class MnistMLP(chainer.Chain):

    # モデルパラメータ (重み行列やバイアスベクトル) を定義しよう
    # Link にはそういったモデルパラメータを内包するクラスが用意されて
    # いるんだ
    # L.Linear は，線形変換 Wx+b を表す Link だ

    def __init__(self, n_in, n_hid, n_out):
        super(MnistMLP, self).__init__(
            l1=L.Linear(n_in, n_hid),
            l2=L.Linear(n_hid, n_hid),
            l3=L.Linear(n_hid, n_out),
        )

    # MnistMLP クラスが呼ばれたらこの計算をするよ
    # Function には活性化関数や損失関数のようなモデルパラメータを
    # 伴わないクラスが用意されているんだ
    # F.sigmoid は，シグモイド関数 1 / (1 + exp(-x)) を表す Function だ

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        return self.l3(h2)

## end [1-3]

if __name__ == "__main__":

    ## 設定変数を決めよう

    # 使用する GPU の ID を設定しよう
    gpu=0

    # ネットワーク各層のユニット数を設定しよう
    n_in = 286
    n_hid = 1024
    n_out = 35

    # 最適化に関するハイパパラメータを設定しよう
    lr = 0.1
    batchsize = 256
    n_epoch = 20

    ##

    # 使用するライブラリを設定しよう (numpy : CPU 用, cupy : GPU 用) 

    if gpu >=0:
        xp = cupy
    else:
        xp = numpy

    ## begin [1-2]
    # 学習データと評価用データを準備しよう
    # numpy の array を使って表現するんだ
    # .npy 形式で保存しておけば読み込みは簡単だよ
    # shape は (データ数，データ毎の次元数)，各特徴ベクトルが縦方向に並ぶ
    # 形式にしておこう

    train_dat=numpy.load("data/train_dat.npy")
    train_lab=numpy.load("data/train_lab.npy")
    N_train  =train_lab.size

    test_dat=numpy.load("data/test_dat.npy")
    test_lab=numpy.load("data/test_lab.npy")
    N_test  =test_lab.size

    ## end [1-2]

    ## begin [1-3]
    # モデルを定義しよう

    model = MnistMLP(n_in, n_hid, n_out)

    # GPU を使う場合は，モデルを GPU 上のメモリへ送ろう
    # 変数 gpu には，使用したい GPU の ID (0,1,...) を指定しよう
    # マシンに搭載されている GPU の情報 (ID を含む) は，nvidia-smi コマンドで確認出来るよ

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    ## end [1-3]

    ## begin [1-4]
    # 最適化方法を設定しよう

    optimizer = optimizers.SGD(lr=lr)
    optimizer.setup(model)

    ## end [1-4]

    ## begin [2-6]
    # 準備は整った
    # 学習用データを使ってモデルの学習をしよう
    # 学習ループを回すんだ

    for epoch in range(1, n_epoch + 1):
        print('epoch', epoch)

        # データのシャッフリングは重要だよ

        perm = numpy.random.permutation(N_train)

        sum_loss = 0.0
        sum_accuracy = 0.0
        for i in range(0, N_train, batchsize):

            ## begin [2-1]
            # 入力データとその教師ラベルのペアを用意しよう
            # Link や Function にあるクラスの入力や出力は，全て chainer.Variable
            # になるんだ
            # だから外部から入力するデータ (入力特徴行列 x と教師信号ベクトル t)
            # も chainer.Variable に格納しておこう
            # CPU なら numpy を，GPU なら cupy を使うんだ
            # それぞれ対応するデバイスのメモリ上にデータを配置してくれるよ

            x = chainer.Variable(xp.asarray(train_dat[perm[i:i + batchsize]]))
            t = chainer.Variable(xp.asarray(train_lab[perm[i:i + batchsize]]))

            ## end [2-1]

            ## begin [2-2]
            # gradient を初期化しよう

            model.zerograds()
            
            ## end [2-2]

            ## begin [2-3]
            # 損失を定義しよう
            # (注) F.softmax_cross_entropy は内部で softmax 関数を
            # 適用した後に cross_entropy を計算するので，model の
            # 側で softmax を掛けておく必要はないぞ

            y = model(x)
            loss = F.softmax_cross_entropy(y,t)

            ## end [2-3]

            ## begin [2-4]
            # gradient を計算しよう

            loss.backward()

            ## end [2-4]

            ## begin [2-5]
            # パラメータを更新しよう

            optimizer.update()
            
            ## end [2-5]

            # 認識率を計算しておこう

            accuracy = F.accuracy(y,t)
            
            # 損失と認識率を累積しよう
            # F.softmax_cross_entropy と F.accuracy は内部処理として，
            # 入力サンプル数での除算 (平均化) をしているんだ
            # ここでは，全サンプル数での loss と accuracy を計算するために，
            # 入力サンプル数で乗算して平均化の効果を打ち消しているよ

            sum_loss += float(loss.data) * len(t.data)
            sum_accuracy += float(accuracy.data) * len(t.data)

        # 学習の進捗を確認するため，エポック毎の損失と認識率を確認しておこう

        print('train mean loss={}, accuracy={}'.format(
            sum_loss / N_train, sum_accuracy / N_train))

    print('')

    ## end [2-5]

    ## begin [3-1]
    # 学習は終わった
    # 評価用データを使ってモデルを評価しよう

    sum_loss = 0.0
    sum_accuracy = 0.0
    for i in range(0, N_test, batchsize):

        # volatile を on に設定しておくと計算グラフを作らないからメモリが節約出来るよ
        # 計算グラフは，学習の際に誤差逆伝播を行うために必要なものだったから，
        # 評価の際には必要ないんだ

        x = chainer.Variable(xp.asarray(test_dat[i:i + batchsize]),
                             volatile='on')
        t = chainer.Variable(xp.asarray(test_lab[i:i + batchsize]),
                             volatile='on')

        # 損失と認識率を累積しよう

        y = model(x)
        
        loss = F.softmax_cross_entropy(y,t)
        sum_loss += float(loss.data) * len(t.data)

        accuracy = F.accuracy(y,t)
        sum_accuracy += float(accuracy.data) * len(t.data)

    # これが学習の結果得られたモデルの評価結果だ
    # 評価データに対する損失と認識率はどうなったかな？

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))

    ## end [3-1]
