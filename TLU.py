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

    def beginLearn():
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
            size0, size1 = len(data[0]), len(data[1])
            class0 = np.transpose(np.asarray(data[0], np.float32))
            class1 = np.transpose(np.asarray(data[1], np.float32))
            mean_x = int((sum(class0[0]) / size0 + sum(class1[0]) / size1) /2)
            mean_y = int((sum(class0[1]) / size0 + sum(class1[1]) / size1) /2)
            threshold = mean_x + mean_y
            print('(mean_x, mean_y, threshold) : (', mean_x, ',', mean_y, ',', threshold, ')')
            x, y = in_normalize(mean_x, mean_y, 250)
            drawPoint(x, y, color='red')

            # prepare training data
            train_X = np.asarray(inputs, np.float32)
            train_Y = np.asarray(results, np.float32).reshape([1, len(results)])

            # create tf model start
            X = tf.placeholder(tf.float32, shape=[None, 2])
            W = tf.Variable(tf.ones([1,2],tf.float32))
            T = tf.constant(threshold, tf.float32, shape=[1, len(results)])
            result = tf.matmul(W, tf.transpose(X))

            # activation function
            Y_ = tf.where(tf.greater_equal(result, T), tf.ones(tf.shape(result), tf.float32),
                                                tf.zeros(tf.shape(result), tf.float32))
            #Y_ = tf.cond(tf.greater_equal(result, T), lambda : tf.constant(1.0), lambda : tf.constant(0.0))
            Y = tf.placeholder(tf.float32, shape=[1, len(results)])

            # misclassfication function
            loss = tf.reduce_mean(tf.abs(Y_-Y) * tf.matmul(W,tf.transpose(X)) - T)
            offset = tf.reshape(tf.reduce_mean(tf.matmul(Y_-Y, X), 0), [1,2])


            optimizer = tf.train.GradientDescentOptimizer(rate)
            train = optimizer.minimize(loss)
            #train = tf.assign_add(W, tf.scalar_mul(rate, offset))
            init = tf.global_variables_initializer()

            slope = -1.0
            w = None
            with tf.Session() as sess:
                sess.run(init)

                for t in range(time):
                    if t % 20 == 0:
                        print(sess.run(W))
                        sess.run(train, feed_dict={X: train_X, Y: train_Y})
                    w = sess.run(W)
                    m = -w[0][0] / w[0][1]
                    #if m - slope < 0.02:
                     #   print('learning time : ', t + 1)
                     #   break
                    slope = m
            print(w[0][0], w[0][1], -w[0][0]/w[0][1])
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

    def drawPoint(x,y,color):
        size = 5
        x1, y1 = x - size, y - size
        x2, y2 = x + size, y + size
        canvas.create_rectangle(x1, y1, x2, y2, fill=color)

    def reset():
        nonlocal inputs,results,data
        canvas.delete('all')
        canvas.create_line(250, 0, 250, 500)
        canvas.create_line(0, 250, 500, 250)
        data = [[], []]
        inputs=[]
        results=[]

    def getData(x, y):
        results.append(current)
        a = normalize(x, y, 250)
        data[current].append(a)
        inputs.append(a)
        print('(', a[0], ',', a[1], ',', current, ')')

    def normalize(x, y, offset):
        return [x-offset, offset-y]

    def in_normalize(x,y,offset):
        return x+offset, offset-y
    #---------------init ui start------------------
    win = Tk()
    win.title('TLU')
    win.geometry('800x600')

    frame1 = Frame(win)
    frame1.pack(side='left')

    canvas = Canvas(frame1, bg='white', width=501, height=501)
    canvas.grid(row=0,column=0,padx=10,pady=10)
    reset()
    canvas.bind('<Button-1>', addData)

    btnLearn = Button(frame1, text='開始學習!', font=(12), height=2, bd=5, command=beginLearn)
    btnLearn.grid(row=1, column=0, padx=10, sticky=W + E)
    # btnLearn.place(x=520, y=520)

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
    txtInput2.insert(INSERT,'1000')
    txtInput2.pack(padx=10, pady=10)

    btnReset = Button(frame2, font=12, width=35, height=12, bd=5, text='重設', command=reset)
    btnReset.pack(padx=10,pady=(250,10))
    #--------------init ui end------------------
    win.mainloop()

if __name__=='__main__':
    main()