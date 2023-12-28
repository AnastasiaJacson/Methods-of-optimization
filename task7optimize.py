import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import numpy.linalg as ln
import numdifftools as nd 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import minimize
from PIL import Image, ImageTk
from matplotlib import cm

calls = 0

class NotEnteredValueError(Exception):
    def __init__(self, value_name):
        self.value_name = value_name


def golden_section_search(f, a, b, tol=1e-5):
    gr = (np.sqrt(5) + 1) / 2 - 1
    c = b - gr * (b - a)
    d = a + gr * (b - a)

    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c

        c = b - gr * (b - a)
        d = a + gr * (b - a)

    return (b + a) / 2


def bfgs_method(x0, epsilon, K, f):
    global calls

    x = x0
    xk = [x]
    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I
    f_x = None
    pk = None
#################
    calls += 2
    gradient = nd.Gradient(f)(x)

    while ln.norm(gradient) > epsilon and k < K:
        print('gradient', gradient, ln.norm(gradient))
        print('x', x)
        f_x = f(x)
    
        pk = -np.dot(Hk, gradient)

        objective_function = lambda alpha: f(x + alpha * pk)
        alpha_k = golden_section_search(objective_function, 0, 1, tol=1e-6)

        xkp1 = x + alpha_k * pk
        sk = xkp1 - x
        x = xkp1

        calls += 2
        fderxp1 = nd.Gradient(f)(xkp1)
        yk = fderxp1 - gradient
        gradient = fderxp1


        ro = 1.0 / (np.dot(yk, sk))
        A1 = I - ro * sk[:, np.newaxis] * yk[np.newaxis, :]
        A2 = I - ro * yk[:, np.newaxis] * sk[np.newaxis, :]
        Hk = np.dot(A1, np.dot(Hk, A2)) + (ro * sk[:, np.newaxis] * sk[np.newaxis, :])

        xk.append(x)
        k += 1
        
    if f_x is not None:
        return x, f_x, np.linalg.norm(pk), k, xk
    else: return (x, )


def newton_method(epsilon, K, x0, f):
    global calls

    x = x0
    xk = [x]
    k = 0

    while k < K:
        calls += 2
        gradient = nd.Gradient(f)(x)
        calls += 3
        hessian = np.array([[2, 0], [0, 2]])
        p = -np.linalg.solve(hessian, gradient)

        if np.linalg.norm(p) < epsilon:
            break

        objective_function = lambda alpha: f(x + alpha * p)
        alpha_k = golden_section_search(objective_function, 0, 1, tol=1e-6)

        x = x + alpha_k * p

        xk.append(x)
        k += 1

    return x, f(x), np.linalg.norm(p), k, xk


def steepest_descent(f, x0, epsilon, K):  
    global calls

    x = x0
    xk = [x]
    k = 0

    while True:
        calls += 2
        gradient = -nd.Gradient(f)(x)

        if np.linalg.norm(gradient) < epsilon or k >= K:
            break

        alpha = golden_section_search(lambda a: f(x + a * gradient), 0, 1, tol=1e-6)
        x = x + alpha * gradient

        xk.append(x)
        k += 1

    return x, f(x), np.linalg.norm(gradient), k, xk


def penalty_function(x, c, f, g, p):
    global calls

    calls += 1
    penalty = f(x) + c / 2 * sum([max(0, gi(x))**p  for gi in g])
    return penalty

def constrained_optimization(f, g, x0, c0, p, epsilon, optim_method, K):
    x = x0
    c = c0
    k = 0
    k_penalty = 0
    prev = float('inf')
    xk = []
    
    while True:
        print('c =', c)
        penalty_func = lambda x: penalty_function(x, c, f, g, p)

        result = optim_method(x0=x, epsilon=epsilon, K=K, f=penalty_func)

        x = result[0]
        if len(result) == 5:
            k += result[3]
            xk += result[4]

        if (all([gi(x) <= 0 for gi in g]) and (abs(prev - penalty_func(x))) < epsilon) or (k >= K) :
            break
        
        c *= 10
        prev = penalty_func(x)
        k_penalty += 1
    
    return x, xk, f(x), penalty_func(x), k, k_penalty


class Optimization:
    def __init__(self, master):
        self.master = master
        master.title("Task 7 desktop app Anastasiia Vynnychuk")

        # Variables
        self.function_str= tk.StringVar(master, value='(x[0] - 2 )**2 + (x[1] - 4)**2')
        self.initial_point = tk.StringVar(master, value='[100, 100]')
        self.precision = tk.StringVar(master, value='1e-6')
        self.max_iterations = tk.IntVar(master, value=1000)
        self.method_var = tk.StringVar(master)
        self.result_label = tk.StringVar(master, value='')
        self.figure = plt.figure(figsize=(7, 7))
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.constraint1_str = tk.StringVar(master, value='2*x[0] + x[1] - 8')
        self.constraint2_str = tk.StringVar(master, value='2*x[0] + 5*x[1]  - 30')

        # GUI Elements
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self.master, text="Функція:").grid(row=1, column=0, padx=10, pady=5)
        ttk.Entry(self.master, textvariable=self.function_str, width=30).grid(row=1, column=1, padx=10, pady=5)

        ttk.Label(self.master, text="Початкова точка:").grid(row=2, column=0, padx=10, pady=5)
        ttk.Entry(self.master, textvariable=self.initial_point, width=30).grid(row=2, column=1, padx=10, pady=5)

        ttk.Label(self.master, text="Точність:").grid(row=3, column=0, padx=10, pady=5)
        ttk.Entry(self.master, textvariable=self.precision, width=30).grid(row=3, column=1, padx=10, pady=5)

        ttk.Label(self.master, text="Максимальна кількість ітерацій:").grid(row=4, column=0, padx=10, pady=5)
        ttk.Entry(self.master, textvariable=self.max_iterations, width=30).grid(row=4, column=1, padx=10, pady=5)

        ttk.Label(self.master, text="Метод оптимізації:").grid(row=5, column=0, padx=10, pady=5)
        ttk.OptionMenu(self.master, self.method_var, "Обрати", "Градієнтний", "Ньютонівський", "Квазі-Ньютонівський").grid(row=5, column=1, padx=10, pady=5)

        ttk.Label(self.master, text="Обмеження:").grid(row=6, column=0, pady=10, padx=5)
        ttk.Entry(self.master, textvariable=self.constraint1_str, width=40).grid(row=6, column=1, columnspan=2, pady=5, padx=5, sticky="ew")
        ttk.Entry(self.master, textvariable=self.constraint2_str, width=40).grid(row=7, column=1, columnspan=2, pady=5, padx=5, sticky="ew")

        ttk.Button(self.master, text="Запустити оптимізацію", command=self.run_optimization).grid(row=8, columnspan=2, column=0, pady=10)

        ttk.Label(self.master, text="Результати:").grid(row=9, column=0, columnspan=2, pady=5)
        ttk.Label(self.master, textvariable=self.result_label).grid(row=10, column=0, columnspan=2, pady=5)
        
        self.canvas_widget.grid(row=0, column=3, rowspan=10, pady=5)

        ttk.Button(self.master, text='Про автора', command=self.show_about).grid(row=1, column=3)

    def show_about(self):
        avatar_path = "C:\\Users\\Admin\\Optimization\\photo_2023-11-28_20-50-01.jpg"
        newWindow = tk.Toplevel(self.master)
 
        # sets the title of the
        newWindow.title("About author")
    
        # A Label widget to show in toplevel
        ttk.Label(newWindow, text ="Author: Anastasiia Vynnychuk\nstudent 4-th grade of Applied Mathematics and Informatics\nLviv national university of Ivan Franko").pack()

        i = Image.open(avatar_path).resize((200, 250))
        img = ImageTk.PhotoImage(i)
        canvas = tk.Canvas(newWindow, width=i.width, height=i.height, borderwidth=0,highlightthickness=0)
        canvas.pack()
        canvas.create_image(0, 0, image=img, anchor=tk.NW)
        newWindow.mainloop()

    def run_optimization(self):
        global calls

        try:
            # Отримання введених даних
            function_str = self.function_str.get()
            if function_str == '':
                raise NotEnteredValueError('функція')
            initial_point = self.initial_point.get()
            precision = self.precision.get()
            max_iterations = self.max_iterations.get()
            method = self.method_var.get()
            constraint1_str = self.constraint1_str.get()
            constraint2_str = self.constraint2_str.get()

            g = [eval(f'lambda x: {constraint1_str}'), eval(f'lambda x: {constraint2_str}'), lambda x: -x[0], lambda x: -x[1]]

            f = eval(f'lambda x: {function_str}')
            initial_point = eval(initial_point)
            epsilon = eval(precision)

            calls = 0

            if method == 'Градієнтний':
                optim_method = steepest_descent
            elif method == 'Ньютонівський':
                optim_method = newton_method
            elif method == 'Квазі-Ньютонівський':
                optim_method = bfgs_method
            
            else:
                raise NotEnteredValueError('метод')
        except NotEnteredValueError as e:
            messagebox.showerror("Бракує вхідних даних", f"Вибери {e.value_name}")
            return
        
        x, xk, f_x, penalty_x, k, k_penalty = constrained_optimization(f=f, g=g, x0=initial_point, c0=1, p=2, epsilon=epsilon, optim_method=optim_method, K=max_iterations)

        # Приклад виводу результатів
        result_str = f"Точка мінімуму: {x[0]:.3f}, {x[1]:.3f}\n"
        result_str += f"Значення функції: {f_x}\n"
        result_str += f"Значення функції пенальті: {penalty_x}\n"
        result_str += f"Кількість ітерацій: {k}\n"
        result_str += f"Кількість ітерацій пенальті: {k_penalty}\n"
        result_str += f"Кількість обчислень цільової функції: {calls}\n"
        self.result_label.set(result_str)

        self.plot_graph(f, x, xk)

    def plot_graph(self, f, x_min, xk):
        plt.clf()
        self.ax = self.figure.add_subplot(111, projection='3d')

        # plot a 3D surface like in the example mplot3d/surface3d_demo
        f2 = lambda x, y: f([x, y])
        X = np.linspace(-10, 10, 100)
        Y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(X, Y)
        
        Z = f2(X, Y)
        self.ax.scatter(x_min[0], x_min[1], f(x_min), c='red', marker='*')
        for xki in xk[:-1]:
            self.ax.scatter(xki[0], xki[1], f(xki), c='green', marker='.')
        xk_points = [[xki[0], xki[1], f(xki)] for xki in xk[:-1]]
        xk_points = np.array(xk_points)
        self.ax.plot(xk_points[:,0], xk_points[:,1], xk_points[:,2])

        surf = self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False, alpha=.3)
        self.figure.colorbar(surf, shrink=0.5, aspect=10)

        self.canvas.draw()

if __name__ == "__main__":
    print("Початок програми")

    try:
        root = tk.Tk()
        app = Optimization(root)
        root.mainloop()
    except Exception as e:
        print(e)
