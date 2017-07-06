from tkinter import *
import tensorflow as tf
import numpy as np

def main():
    current = 0
    colors = ['lime', 'cyan']

    # input record
    data = [[], []]
    inputs = []
    results = []

    def beginLearn(mode):
        print('click')
        try:
            # learning rate
            rate = float(txtInput1.get(1.0,END))
            # learning time
            time = int(txtInput2.get(1.0,END))
        except:
            print('parsing error !')
        else:
            print(rate, ',', time)
            # optimize threshold start
            dist = 500000
            mean_x, mean_y = 0, 0
            for data0 in data[0]:
                for data1 in data[1]:
                    x1, y1 = data0[0], data0[1]
                    x2, y2 = data1[0], data1[1]
                    tmp = (x1 - x2) ** 2 + (y1 - y2) ** 2
                    if tmp < dist:
                        dist = tmp
                        mean_x = int((x1 + x2) / 2)
                        mean_y = int((y1 + y2) / 2)
            threshold = mean_x + mean_y
            print('(mean_x, mean_y, threshold) : (', mean_x, ',', mean_y, ',', threshold, ')')
            x, y = in_normalize(mean_x, mean_y, 250)
            drawPoint(x, y, color='red')

            # prepare training data
            train_X = np.asarray(inputs, np.float32)
            train_Y = np.asarray(results, np.float32).reshape([1, len(results)])

            # create tf model start
            X = tf.placeholder(tf.float32, shape=[len(results), 2])
            W = tf.Variable(tf.ones([1,2],tf.float32))
            T = tf.constant(threshold, tf.float32, shape=[1, len(results)])
            result = tf.matmul(W, tf.transpose(X))

            # activation function
            Y_ = tf.where(tf.greater_equal(result, T), tf.ones(tf.shape(result), tf.float32),
                                                tf.zeros(tf.shape(result), tf.float32))
            Y = tf.placeholder(tf.float32, shape=[1, len(results)])

            # misclassfication function -> define loss function

            if mode == 1:
                print(mode)
                loss = tf.reduce_mean(tf.abs(Y - Y_) * tf.matmul(W, tf.transpose(X)) - T)
                optimizer = tf.train.AdamOptimizer(rate)
            elif mode == 2:
                loss = tf.reduce_mean(tf.matmul(Y - Y_, tf.transpose(tf.matmul(W, tf.transpose(X)) - T)))
                optimizer = tf.train.AdamOptimizer(rate)

            else:
                loss = tf.reduce_mean(tf.matmul(Y_ - Y, tf.transpose(tf.matmul(W, tf.transpose(X)) - T)))
                optimizer = tf.train.GradientDescentOptimizer(rate)


            if mode == 0:
                offset = tf.reduce_mean(tf.matmul(Y - Y_, X), 0, keep_dims=True)
                train = tf.assign_add(W, tf.scalar_mul(rate, offset))
            else:
                train = optimizer.minimize(loss)
            init = tf.global_variables_initializer() # very important

            # create tf model end
            with tf.Session() as sess:
                sess.run(init)
                print(sess.run(W))
                for t in range(time):

                    #if loss1 == 0.0:
                    #    print('learning time : ', t)
                    #    break
                    sess.run(train, feed_dict={X: train_X, Y: train_Y})
                    if t % 100 == 0:
                        w = sess.run(W)
                        print(w, -w[0][0] / w[0][1])
                        loss1 = sess.run(train, feed_dict={X: train_X, Y: train_Y})
                        print('loss : ', loss1)
                        drawLinearFunction(w[0][0], w[0][1], mean_x, mean_y, '#CCC')
                w = sess.run(W)
            drawAxis()
            drawData(0)
            drawData(1)
            drawPoint(x, y, color='red')
            drawLinearFunction(w[0][0], w[0][1], mean_x, mean_y, 'red')

    def changeColor():
        nonlocal current
        current = (current + 1) % 2
        color = colors[current]
        btnChange["bg"] = color

    def addData(event):
        getData(event.x, event.y)
        drawPoint(event.x, event.y, colors[current])

    def drawLinearFunction(w0, w1, mean_x, mean_y, color):
        x1 = -250
        y1 = -(w0 * (x1 - mean_x)) / w1 + mean_y
        x2 = 250
        y2 = -(w0 * (x2 - mean_x)) / w1 + mean_y
        x1, y1 = in_normalize(x1, y1, 250)
        x2, y2 = in_normalize(x2, y2, 250)
        canvas.create_line(x1, y1, x2, y2, fill=color)

    def drawData(class_num):
        color = colors[class_num]
        for point in data[class_num]:
            x, y = in_normalize(point[0], point[1], 250)
            drawPoint(x, y, color)

    def drawPoint(x,y,color):
        size = 5
        x1, y1 = x - size, y - size
        x2, y2 = x + size, y + size
        canvas.create_rectangle(x1, y1, x2, y2, fill=color)

    def drawAxis():
        canvas.create_line(250, 0, 250, 500)
        canvas.create_line(0, 250, 500, 250)

    def reset(mode):
        nonlocal inputs,results,data
        canvas.delete('all')
        drawAxis()
        if mode == 0:
            data = [[], []]
            inputs = []
            results = []
        else:
            drawData(0)
            drawData(1)

    def getData(x, y):
        results.append(current)
        a = normalize(x, y, 250)
        data[current].append(a)
        inputs.append(a)
        print('(', a[0], ',', a[1], ',', current, ')')

    def normalize(x, y, offset):
        return [x-offset, offset-y]

    def in_normalize(x, y, offset):
        return x+offset, offset-y
    #---------------init ui start------------------
    win = Tk()
    win.title('TLU')
    win.geometry('800x600')

    frame1 = Frame(win)
    frame1.pack(side='left')

    canvas = Canvas(frame1, bg='white', width=501, height=501)
    canvas.grid(row=0, column=0, padx=10, pady=10)
    reset(0)
    canvas.bind('<Button-1>', addData)

    frame2 = Frame(win)
    frame2.pack()

    btnChange = Button(frame2, text='換顏色', width=35, height=1, bg=colors[current], command=changeColor)
    btnChange.pack(padx=10,pady=12)
    #btnChange.place(x=520, y=10)

    lbl1 = Label(frame2, text='學習率(0<a<=1):', font=12)
    lbl1.pack(pady=10)

    txtInput1 = Text(frame2,font=12,width=35,height=1)
    txtInput1.insert(INSERT,'0.001')
    txtInput1.pack(padx=10,pady=10)

    lbl2 = Label(frame2, text='學習次數:', font=12)
    lbl2.pack(pady=10)

    txtInput2 = Text(frame2, font=12, width=35, height=1)
    txtInput2.insert(INSERT,'50000')
    txtInput2.pack(padx=10, pady=10)

    frame3 = Frame(frame2)
    frame3.pack(padx = 10, pady = (15, 10))

    btnLearn1 = Button(frame3, font=12, width=35, height=2, bd=5, text='學習1',
                       command=lambda :beginLearn(0))
    btnLearn1.pack(pady=(0,10))

    frame4 = Frame(frame3)
    frame4.pack(pady=(0, 10))

    btnLearn2 = Button(frame4, font=12, width=5, height=2, bd=5, text='學習2',
                       command=lambda :beginLearn(1))
    btnLearn2.pack(side=LEFT, padx=(0, 3))

    btnLearn3 = Button(frame4, font=12, width=5, height=2, bd=5, text='學習3',
                       command=lambda: beginLearn(2))
    btnLearn3.pack(side=LEFT, padx=(0, 3))

    btnLearn4 = Button(frame4, font=12, width=5, height=2, bd=5, text='學習4',
                       command=lambda: beginLearn(3))
    btnLearn4.pack(side=LEFT)

    btnReset1 = Button(frame3, font=12, width=35, height=2, bd=5, text='重設(資料維持)',
                       command=lambda :reset(1))
    btnReset1.pack(side=TOP, pady=(0, 10))

    btnReset2 = Button(frame3, font=12, width=35, height=2, bd=5, text='重設',
                       command=lambda :reset(0))
    btnReset2.pack(side=TOP, pady=(0,10))
    #--------------init ui end------------------
    win.mainloop()

if __name__=='__main__':
    main()