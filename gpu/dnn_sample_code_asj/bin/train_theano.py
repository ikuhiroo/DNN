#!/usr/bin/env python
#-*- coding: utf-8 -*-

## begin [1-1]
# 必要なモジュールをインポートしよう

from __future__ import print_function

import numpy

import theano
import theano.tensor as T

## end [1-1]

## begin [1-4]の準備
# モデルを定義しよう
# Chainer における L.Linear に相当するモデルを Theano では自分で実装するよ

class Linear(object):

    # モデルパラメータ (重み行列やバイアスベクトル) を定義しよう

    def __init__(self, n_in, n_out):

        # モデルパラメータの初期値を設定

        W_values = numpy.asarray(numpy.random.normal(
            0, numpy.sqrt(1. / n_in), (n_in, n_out)),
            dtype=theano.config.floatX)
        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        
        # Theano ではモデルパラメータ (重み行列やバイアスベクトル)
        # は theano.shared を用いて自分で明示的に設定するよ

        self.W = theano.shared(value=W_values, name='W')
        self.b = theano.shared(value=b_values, name='b')

    # Linear クラスが呼ばれたらこの計算をする数式を返すよ

    def __call__(self, x):
        return T.dot(x, self.W) + self.b

# MLP モデルを定義しよう

class MnistMLP(object):

    # モデルパラメータ (重み行列やバイアスベクトル) を定義しよう
    # 上で定義した Linear モデルを使うんだ

    def __init__(self, n_in, n_hid, n_out):
        self.l1=Linear(n_in, n_hid)
        self.l2=Linear(n_hid, n_hid)
        self.l3=Linear(n_hid, n_out)

    # MnistMLP クラスが呼ばれたらこの計算をする数式を返すよ
    # T.nnet には活性化関数や損失関数といった基本的な数式が
    # 用意されているんだ
    # T.nnet は，シグモイド関数 1 / (1 + exp(-x)) を表す数式だ

    def __call__(self, x):
        h1 = T.nnet.sigmoid(self.l1(x))
        h2 = T.nnet.sigmoid(self.l2(h1))
        return T.nnet.softmax(self.l3(h2))

## end [1-4]

if __name__ == "__main__":

    ## 設定変数を決めよう

    # ネットワーク各層のユニット数を設定しよう
    n_in = 286
    n_hid = 1024
    n_out = 35

    # 最適化に関するハイパパラメータを設定しよう
    lr = 0.1
    batchsize = 256
    n_epoch = 20

    ##

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
    # 外部から入力するシンボル (入力変数) を定義しよう
    # ここでは，入力特徴行列 x と教師信号ベクトル t の準備をするよ

    x = T.fmatrix('x')
    t = T.ivector('t')

    ## end [1-3]

    ## begin [1-4]の本体
    # モデルを定義しよう

    model = MnistMLP(n_in, n_hid, n_out)

    # ネットワーク出力を計算する数式を作成するよ

    y = model(x)

    ## end [1-4]

    # 学習用 Theano 関数 (update) を作成しよう

    ## begin [1-5]
    # loss を計算する数式を作成するよ

    equ_loss = T.mean(T.nnet.categorical_crossentropy(y,t))

    ## end [1-5]

    ## begin [1-6]
    # gradient を計算する数式を作成しよう
    # Theano には自動微分という便利な機能が実装されていて，T.grad を使えば
    # loss を対応するモデルパラメータで微分した結果の数式を自動的に作ってくれるよ

    equ_grad = T.grad(equ_loss,[model.l1.W, model.l1.b,
                                model.l2.W, model.l2.b,
                                model.l3.W, model.l3.b])

    ## end [1-6]

    ## begin [1-7]
    # 最適化を行う数式を作成しよう
    # ここでは最も基本的な最適化手法である SGD を実装してみるよ
    # Chainer とは違って最適化方法の部分を自分で実装する必要はあるけど，
    # Theano はその分だけより柔軟に実装出来るぞ

    equ_update=[(model.l1.W, model.l1.W - lr*equ_grad[0]),
                (model.l1.b, model.l1.b - lr*equ_grad[1]),
                (model.l2.W, model.l2.W - lr*equ_grad[2]),
                (model.l2.b, model.l2.b - lr*equ_grad[3]),
                (model.l3.W, model.l3.W - lr*equ_grad[4]),
                (model.l3.b, model.l3.b - lr*equ_grad[5])]

    ## end [1-7]

    ## begin [1-8]
    # 数式を作成しただけでは使えないんだ
    # theano.function を使うと数式を実際に計算するコードを自動的に
    # 作成してくれるよ
    # GPU モードで起動すると，この数式の計算コードが GPU コードで作成
    # されるから計算がすごく早くなるんだ

    func_update=theano.function(
        inputs=[x,t],outputs=equ_loss,updates=equ_update)

    ## end [1-8]

    ## begin [3-1]の準備
    # 評価用 Theano 関数 (loss + accuracy) を作成しよう
    # ここで，equ_acc の中身について説明しておくよ
    # (1) T.argmax によって，ネットワーク出力 y (n_batch × n_out) の各行に対して，
    #     最も大きい要素を持つインデックス(クラス番号)を出力するよ
    # (2) T.neq によって，ネットワークの出力結果と教師信号のクラス番号を要素毎に比較し，
    #     変数が異なる場合には 1 を，同じ場合には 0 を出力するよ
    # (3) T.mean によって，T.neq の出力結果を平均することで誤り率を計算するよ
    # (4) 1 から減算することで，accuracy (正解率) へと変換しているよ

    func_loss=theano.function(inputs=[x,t],outputs=equ_loss)

    equ_acc=(1.0-T.mean(T.neq(T.argmax(y,axis=1),t)))

    func_acc=theano.function(inputs=[x,t],outputs=equ_acc)

    ## end [3-1]

    ## begin [2-2]
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

            x = numpy.asarray(train_dat[perm[i:i + batchsize]])
            t = numpy.asarray(train_lab[perm[i:i + batchsize]])

            # パラメータを更新しよう + 損失を計算しておこう

            loss = func_update(x,t)

            ## end [2-1]

            # 認識率を計算しておこう

            acc = func_acc(x,t)
            
            # 損失と認識率を累積しよう

            sum_loss += float(loss) * len(t)
            sum_accuracy += float(acc) * len(t)

        # 学習の進捗を確認するため，エポック毎の損失と認識率を確認しておこう

        print('train mean loss={}, accuracy={}'.format(
            sum_loss / N_train, sum_accuracy / N_train))

    print('')

    ## end [2-2]

    ## begin [3-1]の本体
    # 学習は終わった
    # 評価用データを使ってモデルを評価しよう

    sum_loss = 0.0
    sum_accuracy = 0.0
    for i in range(0, N_test, batchsize):

        # 入力データとその教師ラベルのペアを用意しよう

        x = numpy.asarray(test_dat[i:i + batchsize])
        t = numpy.asarray(test_lab[i:i + batchsize])

        # 損失を計算しておこう

        loss = func_loss(x,t)

        # 認識率を計算しておこう

        acc = func_acc(x,t)
            
        # 損失と認識率を累積しよう

        sum_loss += float(loss) * len(t)
        sum_accuracy += float(acc) * len(t)

    # これが学習の結果得られたモデルの評価結果だ
    # 評価データに対する損失と認識率はどうなったかな？

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))

    ## end [3-1]
