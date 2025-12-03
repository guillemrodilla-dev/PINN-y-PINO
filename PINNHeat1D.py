import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Some plot settings --------------------------------
plt.close("all")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 100

# 1. Define de neural network model -----------------
# create_model() - This function defines an returns a dictionary representing a neural network model. Within the dictionary the keys are the layer names and the values are the corresponding Dense Layers.
alpha = 0.01

def create_model():
    model = {
        'dense1': tf.keras.layers.Dense(50, activation='tanh'),
        'dense2': tf.keras.layers.Dense(50, activation='tanh'),
        'dense3': tf.keras.layers.Dense(50, activation='tanh'),
        'output_layer': tf.keras.layers.Dense(1),
    }
    return model

# call_model() - This function defines the forward pass of the neural network. It takes as input a dictionary model (created by create_model()) and an input tensor x.
def call_model(model,x,t):
    X = tf.concat([x, t], axis=1)  # Combina x y t como entrada
    X = model['dense1'](X)
    X = model['dense2'](X)
    X = model['dense3'](X)
    X = model['output_layer'](X)
    return X # Este es el valor de y
    
# model = create_model()
# print(model)

# 2. Define the differential equation using tf.GradientTape ----------------------------------------
def pde(x, t, model, alpha):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x,t])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x,t])
            u = call_model(model, x, t)
        u_x = tape1.gradient(u, x)
        u_t = tape1.gradient(u, t)
    u_xx = tape2.gradient(u_x, x)
    del tape1, tape2
    return u_t - alpha*u_xx

# 3. Definee the loss function -------------------------
def loss(model, x, t, x_bc, t_bc, y_bc, x_ic, t_ic, y_ic):
    # Compute the mean squared error of the residual equation
    res = pde(x, t, model, alpha)
    loss_pde = tf.reduce_mean(tf.square(res))
    # Compute the mean squared error of the boundary conditions
    y_bc_pred = call_model(model, x_bc, t_bc)
    loss_bc = tf.reduce_mean(tf.square(y_bc-y_bc_pred))
    # Compute the mean squared error of the initial conditions
    y_ic_pred = call_model(model, x_ic, t_ic)
    loss_ic = tf.reduce_mean(tf.square(y_ic-y_ic_pred))

    return loss_pde + loss_bc + loss_ic

# 4. Define the training step --------------------------
def train_step(model, x, t, x_bc, t_bc, y_bc, x_ic, t_ic, y_ic, optimizer):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, t, x_bc, t_bc, y_bc, x_ic, t_ic, y_ic)
    grads = tape.gradient(loss_value, [layer.trainable_variables for layer in model.values()])
    # Flatten the list of trainable variables
    grads = [grad for sublist in grads for grad in sublist]
    variables = [var for layer in model.values() for var in layer.trainable_variables]
    optimizer.apply_gradients(zip(grads, variables))
    return loss_value

# 5. Setting up the problem -------------------------

# Generate training data
Nx_tr = 50
Nt_tr = 50

x_tr = np.linspace(0,1,Nx_tr)
t_tr = np.linspace(0,1,Nt_tr)
X_tr,T_tr = np.meshgrid(x_tr,t_tr)

x_train = X_tr.reshape(-1,1)
t_train = T_tr.reshape(-1,1)
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
t_train = tf.convert_to_tensor(t_train, dtype=tf.float32)

# Boundary data
t_bc = np.linspace(0,1,Nt_tr).reshape(-1,1) # (Nt,1) Puntos en el eje t
x_bc = np.vstack([np.zeros_like(t_bc), np.ones_like(t_bc)]) # (2*Nt,1) x = 0 y x = 1 en los puntos de t
t_bc = np.vstack([t_bc, t_bc]) # (2*Nt,1) 
y_bc = np.zeros_like(x_bc) # (2*Nt,1) T = 0 en ambos bordes
t_bc = tf.convert_to_tensor(t_bc, dtype=tf.float32)
x_bc = tf.convert_to_tensor(x_bc, dtype=tf.float32)
y_bc = tf.convert_to_tensor(y_bc, dtype=tf.float32)

# Initial data
x_ic = np.linspace(0,1,Nx_tr).reshape(-1,1)
t_ic = np.zeros_like(x_ic)
# y_ic = np.sin(np.pi*x_ic)
y_ic = np.zeros_like(x_ic)
x_ic = tf.convert_to_tensor(x_ic, dtype=tf.float32)
t_ic = tf.convert_to_tensor(t_ic, dtype=tf.float32)
y_ic = tf.convert_to_tensor(y_ic, dtype=tf.float32)


# Define the PINN model
model = create_model()

# Define the optimizer with a learning rate scheduler
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 1e-3,
    decay_steps = 1000,
    decay_rate = 0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate = lr_scheduler)

# Train the model
epochs = 2000
for epoch in range(epochs):
    loss_value = train_step(model, x_train, t_train, x_bc, t_bc, y_bc, x_ic, t_ic, y_ic, optimizer)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss_value.numpy()}")

# 6. Predict the solution ---------------------------
Nx_te = 100
Nt_te = 100

x_te = np.linspace(0,1,Nx_te)
t_te = np.linspace(0,1,Nt_te)
X_te,T_te = np.meshgrid(x_te,t_te)

x_test = X_te.reshape(-1,1)
t_test = T_te.reshape(-1,1)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
t_test = tf.convert_to_tensor(t_test, dtype=tf.float32)
y_pred = call_model(model, x_test, t_test).numpy().reshape(100,100)

# Analytical solution
y_true  = np.exp(-alpha*np.pi**2*t_test)*np.sin(np.pi*x_test)

# 7. Plot the result --------------------------------
plt.figure(figsize=(6,4))
plt.imshow(y_pred, extent=[0,1,0,1], origin='lower', aspect='auto')
plt.colorbar(label='u_pred(x,t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Predicción PINN para la ecuación del calor')
plt.show()

# =========================
# COMPARACIÓN PINN vs. SOLUCIÓN ANALÍTICA
# =========================

# (1) Error absoluto punto a punto
#error_abs = np.abs(y_pred - y_true.numpy().reshape(Nt_te, Nx_te))

#plt.figure(figsize=(6,4))
#plt.imshow(error_abs, extent=[0,1,0,1], origin='lower', aspect='auto', cmap='inferno')
#plt.colorbar(label='|u_pred - u_true|')
#plt.xlabel('x')
#plt.ylabel('t')
#plt.title('Error absoluto entre PINN y solución analítica')
#plt.show()

# (2) MSE global
#mse = np.mean((y_pred - y_true.numpy().reshape(Nt_te, Nx_te))**2)
#print(f"Error cuadrático medio (MSE): {mse:.2e}")

# (3) Comparación en cortes temporales
#plt.figure(figsize=(6,4))
#for t_val in [0, 0.25, 0.5, 1.0]:
#    idx = int(t_val * (Nt_te - 1))
#    plt.plot(x_te, y_true.numpy().reshape(Nt_te, Nx_te)[idx, :], 'k-', label=f'True t={t_val}')
#    plt.plot(x_te, y_pred[idx, :], '--', label=f'Pred t={t_val}')
#plt.xlabel('x')
#plt.ylabel('u(x,t)')
#plt.title('Comparación de cortes temporales')
#plt.legend()
#plt.show()