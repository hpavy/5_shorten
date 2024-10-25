# Avec Z et le choix des points avec une certaine proba
from deepxrte.geometry import Rectangle
import torch
import torch.nn as nn
import torch.optim as optim
from model import PINNs
from utils import read_csv, write_csv
from train import train
from pathlib import Path
import time
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

time_start = time.time()

############# LES VARIABLES ################

folder_result = "1_first_fonctionne"  # le nom du dossier de résultat

random_seed_train = None
# la seed de test, toujours garder la même pour pouvoir comparer
random_seed_test = 2002


##### Le modèle de résolution de l'équation de la chaleur
nb_itt = 3000  # le nb d'epoch
save_rate = 300
poids = [1, 1]  # les poids pour la loss

batch_size = 5000  # la taille d'un batch
# batch_size_pde = 1  # le nb de points pour la pde ### Pour l'instant on prend la même taille

n_pde = 10000

n_data_test = 5000
n_pde_test = 5000

Re = 3900

lr = 1e-3

gamma_scheduler = 0.999


##### Le code ###############################
###############################################

# La data
# On adimensionne la data
df = pd.read_csv("data.csv")
df_modified = df[
    (df["Points:0"] >= 0.22)
    & (df["Points:1"] >= -0.1)
    & (df["Points:1"] <= 0.1)
    & (df["Time"] > 4)
    & (df["Time"] < 6)
]
# Uniquement la fin de la turbulence

x_full, y_full, t_full = (
    np.array(df_modified["Points:0"]),
    np.array(df_modified["Points:1"]),
    np.array(df_modified["Time"]),
)
u_full, v_full, p_full = (
    np.array(df_modified["Velocity:0"]),
    np.array(df_modified["Velocity:1"]),
    np.array(df_modified["Pressure"]),
)

x_norm_full = (x_full - x_full.mean()) / x_full.std()
y_norm_full = (y_full - y_full.mean()) / y_full.std()
t_norm_full = (t_full - t_full.mean()) / t_full.std()
p_norm_full = (p_full - p_full.mean()) / p_full.std()
u_norm_full = (u_full - u_full.mean()) / u_full.std()
v_norm_full = (v_full - v_full.mean()) / v_full.std()


X_full = np.array([x_norm_full, y_norm_full, t_norm_full], dtype=np.float32).T
U_full = np.array([u_norm_full, v_norm_full, p_norm_full], dtype=np.float32).T
np.random.shuffle(X_full)  # pour pouvoir prendre des éléments au pif

# On divise en un certain nombre de points
masque = X_full[:, 2] == np.unique(t_norm_full)[23]
X_full[masque].shape
couple_x_y = [(x, y) for x, y in X_full[masque][:, :2]]
couple_x_y_reduce = couple_x_y[:80]
masque_reduce = np.isin(X_full[:, :2], couple_x_y_reduce).all(axis=1)
X_reduce = X_full[masque_reduce]
U_reduce = U_full[masque_reduce]

X = torch.from_numpy(X_reduce).requires_grad_().to(device)
U = torch.from_numpy(U_reduce).requires_grad_().to(device)

t_norm_min = t_norm_full.min()
t_norm_max = t_norm_full.max()

x_norm_max = x_norm_full.max()
y_norm_max = y_norm_full.max()
x_norm_min = x_norm_full.min()
y_norm_min = y_norm_full.min()


# On regarde si le dossier existe
dossier = Path(folder_result)
dossier.mkdir(parents=True, exist_ok=True)


rectangle = Rectangle(
    x_max=x_norm_max,
    y_max=y_norm_max,
    t_min=t_norm_min,
    t_max=t_norm_max,
    x_min=x_norm_min,
    y_min=y_norm_min,
)  # le domaine de résolution

X_pde = rectangle.generate_lhs(n_pde).to(device)

# les points initiaux du train
# Les points de pde


### Pour test
torch.manual_seed(random_seed_test)
np.random.seed(random_seed_test)
X_test_pde = rectangle.generate_lhs(n_pde_test).to(device)
points_coloc_test = np.random.choice(len(X_full), n_data_test, replace=False)
X_test_data = torch.from_numpy(X_full[points_coloc_test]).to(device)
U_test_data = torch.from_numpy(U_full[points_coloc_test]).to(device)


# Initialiser le modèle
model = PINNs().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_scheduler)
loss = nn.MSELoss()

# On plot les print dans un fichier texte
with open(folder_result + "/print.txt", "a") as f:
    # On regarde si notre modèle n'existe pas déjà
    if Path(folder_result + "/model_weights.pth").exists():
        # Charger l'état du modèle et de l'optimiseur
        checkpoint = torch.load(folder_result + "/model_weights.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=gamma_scheduler
        )
        print("\nModèle chargé\n", file=f)
        print("\nModèle chargé\n")
        csv_train = read_csv(folder_result + "/train_loss.csv")
        csv_test = read_csv(folder_result + "/test_loss.csv")
        train_loss = {
            "total": list(csv_train["total"]),
            "data": list(csv_train["data"]),
            "pde": list(csv_train["pde"]),
        }
        test_loss = {
            "total": list(csv_test["total"]),
            "data": list(csv_test["data"]),
            "pde": list(csv_test["pde"]),
        }
        print("\nLoss chargée\n", file=f)
        print("\nLoss chargée\n")

    else:
        print("Nouveau modèle\n", file=f)
        print("Nouveau modèle\n")
        train_loss = {"total": [], "data": [], "pde": []}
        test_loss = {"total": [], "data": [], "pde": []}

    if random_seed_train is not None:
        torch.manual_seed(random_seed_train)
        np.random.seed(random_seed_train)
    ######## On entraine le modèle
    ###############################################
    train(
        nb_itt=nb_itt,
        train_loss=train_loss,
        test_loss=test_loss,
        poids=poids,
        model=model,
        loss=loss,
        optimizer=optimizer,
        X=X,
        U=U,
        X_pde=X_pde,
        X_test_pde=X_test_pde,
        X_test_data=X_test_data,
        U_test_data=U_test_data,
        Re=Re,
        time_start=time_start,
        f=f,
        u_mean=u_full.mean(),
        v_mean=v_full.mean(),
        x_std=x_full.std(),
        y_std=y_full.std(),
        t_std=t_full.std(),
        u_std=u_full.std(),
        v_std=v_full.std(),
        p_std=p_full.std(),
        folder_result=folder_result,
        save_rate=save_rate,
        batch_size=batch_size,
        scheduler=scheduler,
    )

####### On save le model et les losses

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    folder_result + "/model_weights.pth",
)
write_csv(train_loss, folder_result, file_name="/train_loss.csv")
write_csv(test_loss, folder_result, file_name="/test_loss.csv")
