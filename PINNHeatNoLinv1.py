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
rho = 7895
Cp = 460

def k(u):
    k0 = 0.3
    beta = 150
    return k0*u + beta

def Q(u):
    gamma = 0.0
    return gamma

def create_model():
    model = {
        'dense1': tf.keras.layers.Dense(100, activation='tanh'),
        'dense2': tf.keras.layers.Dense(100, activation='tanh'),
        'dense3': tf.keras.layers.Dense(100, activation='tanh'),
        'output_layer': tf.keras.layers.Dense(1),
    }
    return model

# call_model() - This function defines the forward pass of the neural network. It takes as input a dictionary model (created by create_model()) and an input tensor x.
def call_model(model,x,y,t):
    X = tf.concat([x,y,t], axis=1)  # Combina x y t como entrada
    X = model['dense1'](X)
    X = model['dense2'](X)
    X = model['dense3'](X)
    X = model['output_layer'](X)
    return X # Este es el valor de y
    
# model = create_model()
# print(model)

# 2. Define the differential equation using tf.GradientTape ----------------------------------------
def pde(x, y, t, model):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x,y,t])

        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x,y,t])
            u = call_model(model, x, y, t)

        u_x = tape1.gradient(u, x)
        u_y = tape1.gradient(u, y)
        u_t = tape1.gradient(u, t)

        k_u = k(u)
        kux = k_u * u_x
        kuy = k_u * u_y

    div_x = tape2.gradient(kux, x)
    div_y = tape2.gradient(kuy, y)

    Q_u = Q(u)

    del tape1, tape2
    return u_t - (div_x + div_y)/(rho*Cp) - Q_u/(rho*Cp)

# 3. Definee the loss function -------------------------
def loss(model, x, y, t, x_bc, y_bc, t_bc, u_bc, x_ic, y_ic, t_ic, u_ic):
    # Compute the mean squared error of the residual equation
    res = pde(x, y, t, model)
    loss_pde = tf.reduce_mean(tf.square(res))
    # Compute the mean squared error of the boundary conditions
    u_bc_pred = call_model(model, x_bc, y_bc, t_bc)
    loss_bc = tf.reduce_mean(tf.square(u_bc-u_bc_pred))
    # Compute the mean squared error of the initial conditions
    u_ic_pred = call_model(model, x_ic, y_ic, t_ic)
    loss_ic = tf.reduce_mean(tf.square(u_ic-u_ic_pred))

    return loss_pde + loss_bc + loss_ic

# 4. Define the training step --------------------------
def train_step(model, x, y, t, x_bc, y_bc, t_bc, u_bc, x_ic, y_ic, t_ic, u_ic, optimizer):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, y, t, x_bc, y_bc, t_bc, u_bc, x_ic, y_ic, t_ic, u_ic)
    grads = tape.gradient(loss_value, [layer.trainable_variables for layer in model.values()])
    # Flatten the list of trainable variables
    grads = [grad for sublist in grads for grad in sublist]
    variables = [var for layer in model.values() for var in layer.trainable_variables]
    optimizer.apply_gradients(zip(grads, variables))
    return loss_value

# 5. Setting up the problem -------------------------

# Generate training data . . . . . . . . . . . . . . . . .
Lx, Ly = 1, 1
Nx, Ny, Nt = 20, 20, 20

x = np.linspace(0,Lx,Nx)
y = np.linspace(0,Ly,Ny)
t = np.linspace(0,1,Nt)

X, Y, T = np.meshgrid(x,y,t, indexing='ij')

x_train = X.reshape(-1,1)
y_train = Y.reshape(-1,1)
t_train = T.reshape(-1,1)

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
t_train = tf.convert_to_tensor(t_train, dtype=tf.float32)

# Boundary data (u(:,:,t) = 0). . . . . . . . . . . . . . . . . . . 

# x = 0
y_bc1, t_bc1 = np.meshgrid(y,t, indexing='ij')
x_bc1 = np.zeros_like(y_bc1)
# x = 1
y_bc2, t_bc2 = np.meshgrid(y,t, indexing='ij')
x_bc2 = np.ones_like(y_bc2)

# y = 0
x_bc3, t_bc3 = np.meshgrid(x,t, indexing='ij')
y_bc3 = np.zeros_like(x_bc3)
# y = 1
x_bc4, t_bc4 = np.meshgrid(x,t, indexing='ij')
y_bc4 = np.ones_like(x_bc4)

x_bc = np.vstack([x_bc1.reshape(-1,1), x_bc2.reshape(-1,1),
                   x_bc3.reshape(-1,1), x_bc4.reshape(-1,1)])
y_bc = np.vstack([y_bc1.reshape(-1,1), y_bc2.reshape(-1,1),
                   y_bc3.reshape(-1,1), y_bc4.reshape(-1,1)])
t_bc = np.vstack([t_bc1.reshape(-1,1), t_bc2.reshape(-1,1),
                   t_bc3.reshape(-1,1), t_bc4.reshape(-1,1)])

u_bc = np.zeros_like(x_bc)

x_bc = tf.constant(x_bc, tf.float32)
y_bc = tf.constant(y_bc, tf.float32)
t_bc = tf.constant(t_bc, tf.float32)
u_bc = tf.constant(u_bc, tf.float32)

# Initial data (u(x,y,0) = 0). . . . . . . . . . . . . . . . . . . 
x_ic, y_ic = np.meshgrid(x,y, indexing='ij')
t_ic = np.zeros_like(x_ic)
u_ic = np.zeros_like(x_ic)


x_ic = tf.constant(x_ic.reshape(-1,1), tf.float32)
y_ic = tf.constant(y_ic.reshape(-1,1), tf.float32)
t_ic = tf.constant(t_ic.reshape(-1,1), tf.float32)
u_ic = tf.constant(u_ic.reshape(-1,1), tf.float32)


# Define the PINN model
model = create_model()

# Define the optimizer 
optimizer = tf.keras.optimizers.Adam(1e-3)


# Train the model
epochs = 2000
for epoch in range(epochs):
    L = train_step(model, x_train, y_train, t_train, 
                   x_bc, y_bc, t_bc, u_bc, 
                   x_ic, y_ic, t_ic, u_ic, 
                   optimizer)
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {L.numpy()}")

# 6. Predict the solution ---------------------------
t0 = 0.5
x_te, y_te = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
t_te = t0*np.ones_like(x_te)

u_pred = call_model(
    model,
    tf.constant(x_te.reshape(-1,1), tf.float32),
    tf.constant(y_te.reshape(-1,1), tf.float32),
    tf.constant(t_te.reshape(-1,1), tf.float32)
).numpy().reshape(100,100)

# 7. Plot the result --------------------------------
plt.figure(figsize=(5,4))
plt.imshow(u_pred, extent=[0,1,0,1], origin='lower', aspect='auto')
plt.colorbar(label='u(x,y,t=0.5)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('PINN 2D - Ecuación del calor')
plt.show()