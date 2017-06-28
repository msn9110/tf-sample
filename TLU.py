from tkinter import *
import tensorflow as tf
import numpy as np

def main():
    current = 0
    colors = ['lime', 'cyan']

    # input record
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
            tmp = np.transpose(np.asarray(inputs))
            mean_x = int(sum(tmp[0] / len(results)))
            mean_y = int(sum(tmp[1] / len(results)))
            threshold = mean_x + mean_y
            x, y = in_normalize(mean_x, mean_y, 250)
            drawPoint(x, y, color='red')
            # prepare training data
            train_X = np.asarray(inputs, np.float32)
            print(train_X[0], train_X[0].shape)

            # create tf model start
            X = tf.placeholder(tf.float32, shape=[1, 2])
            W = tf.Variable(tf.ones([1,2],tf.float32))
            T = tf.constant(threshold, tf.float32, shape=[])
            result = tf.reshape(tf.matmul(W,tf.transpose(X)), [])

            # activation function
            Y_ = tf.cond(tf.greater_equal(result, T), lambda : tf.constant(1), lambda : tf.constant(0))
            Y = tf.placeholder(tf.int32, shape=[])
            # misclassfication function
            loss = tf.cast(tf.subtract(Y, Y_), tf.float32)
            optimizer = tf.train.GradientDescentOptimizer(rate)
            # train = optimizer.minimize(loss)
            train = tf.assign(W, tf.add(W, tf.scalar_mul(tf.multiply(tf.constant(rate), loss), X)))
            init = tf.global_variables_initializer()

            w = None
            with tf.Session() as sess:
                sess.run(init)

                for t in range(time):
                    if t % 20 == 0:
                        print(sess.run(W))
                    for i in range(len(inputs)):
                        input1 = train_X[i].reshape([1,2])
                        #print(sess.run(result, feed_dict={X: input1}))
                        #print('predict : ', sess.run(Y_, feed_dict={X: input1}))
                        #print('loss : ', sess.run(loss, feed_dict={X: input1, Y: results[i]}))
                        sess.run(train, feed_dict={X: input1, Y: results[i]})
                    w = sess.run(W)
            print(w[0][0], w[0][1])
            drawLinearFunction(w[0][0], w[0][1], threshold, 'purple')

    def changeColor():
        nonlocal current
        current=(current+1)%2
        color=colors[current]
        btnChange["bg"]=color

    def addData(event):
        getData(event.x,event.y)
        drawPoint(event.x,event.y,colors[current])

    def drawLinearFunction(w0,w1, threshold, color):
        x1 = -250
        y1 = (threshold - w0 * x1) / w1
        x2 = 250
        y2 = (threshold - w0 * x2) / w1
        x1, y1 = in_normalize(x1, y1, 250)
        x2, y2 = in_normalize(x2, y2, 250)
        canvas.create_line(x1, y1, x2, y2, fill=color)
    def drawPoint(x,y,color):
        size = 7
        x1, y1 = x - size, y - size
        x2, y2 = x + size, y + size
        canvas.create_rectangle(x1, y1, x2, y2, fill=color)

    def reset():
        nonlocal inputs,results
        canvas.delete('all')
        canvas.create_line(250, 0, 250, 500)
        canvas.create_line(0, 250, 500, 250)
        inputs=[]
        results=[]

    def getData(x,y):
        results.append(current)
        a=normalize(x,y,250)
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

    btnLearn = Button(frame1, text='開始學習!', font=(12), height=3, bd=5, command=beginLearn)
    btnLearn.grid(row=1, column=0, padx=10, sticky=W + E)
    # btnLearn.place(x=520, y=520)

    frame2 = Frame(win)
    frame2.pack()

    btnChange = Button(frame2, text='換顏色', width=35, height=1, bg=colors[current], command=changeColor)
    btnChange.pack(padx=10,pady=12)
    #btnChange.place(x=520, y=10)

    lbl1 = Label(frame2, text='學習速率(0<a<=1):', font=12)
    lbl1.pack(pady=10)

    txtInput1 = Text(frame2,font=12,width=35,height=1)
    txtInput1.insert(INSERT,'0.8')
    txtInput1.pack(padx=10,pady=10)

    lbl2 = Label(frame2, text='學習次數:', font=12)
    lbl2.pack(pady=10)

    txtInput2 = Text(frame2, font=12, width=35, height=1)
    txtInput2.insert(INSERT,'100')
    txtInput2.pack(padx=10, pady=10)

    btnReset = Button(frame2, font=12, width=35, height=3, bd=5, text='重設', command=reset)
    btnReset.pack(padx=10,pady=(320,10))
    #--------------init ui end------------------
    win.mainloop()

if __name__=='__main__':
    main()