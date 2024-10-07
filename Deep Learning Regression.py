import importlib
import sys

packages = [
    ('pandas', 'pd'),
    ('geopandas', 'gpd'),
    #('qgrid', None),
    ('ipywidgets', None),
    ('widgetsnbextension', None),
    ('folium', None),
    ('matplotlib.pyplot', 'plt'),
    #('matplotlib.image', 'mpimg'),
    #('mapclassify', None),
    ('numpy', 'np'),
    ('decimal', None),
    ('math', None),
    ('plotly', None),
    ('sklearn', None),
    #('mglearn', None),
    ('seaborn', 'sns'),
    ('fast_ml', None),
    ('scipy', 'stats'),
    #('pandasgui', 'show'),
    ('sklearn.datasets', 'make_classification'),
    #('sklearn.metrics', 'plot_confusion_matrix'),
    ('sklearn.model_selection', 'train_test_split'),
    ('sklearn.svm', 'SVC'),
    ('sklearn.datasets', 'make_regression'),
    ('sklearn.model_selection', 'cross_val_score'),
    ('sklearn.model_selection', 'RepeatedStratifiedKFold'),
    ('sklearn.ensemble', 'RandomForestRegressor'),
    ('sklearn.datasets', 'make_moons'),
    ('sklearn.tree', 'DecisionTreeClassifier'),
    ('sklearn', 'metrics'),
    ('sklearn.model_selection', 'ShuffleSplit'),
    ('sklearn.model_selection', 'RandomizedSearchCV'),
    ('sklearn.model_selection', 'GridSearchCV'),
    ('sklearn.metrics', 'accuracy_score'),
    ('fast_ml.model_development', 'train_valid_test_split'),
    #('tensorflow', 'tf'),
    ('tensorflow.keras', 'keras'),
    ('tensorflow.keras.layers', 'Conv2D'),
    ('tensorflow.keras.layers', 'MaxPooling2D'),
    ('tensorflow.keras.layers', 'Dropout'),
    ('tensorflow.keras.layers', 'Flatten'),
    ('tensorflow.keras.layers', 'Dense'),
    ('tensorflow.keras.models', 'Sequential'),
    ('scikeras.wrappers', 'KerasRegressor'),
    ('sklearn.model_selection', 'KFold'),
    ('datetime', 'datetime'),
    ('datetime', 'date'),
    ('skopt.learning', 'GaussianProcessRegressor'),
    ('skopt.learning', 'RandomForestRegressor'),
    ('skopt.learning', 'ExtraTreesRegressor'),
    ('skopt.learning', 'GradientBoostingQuantileRegressor'),
    ('torch', None),
    ('torch.nn.functional', 'F'),
    ('skopt', 'Optimizer'),
    ('sklearn.metrics', 'r2_score'),
    ('skopt.space', 'Real'),
    ('skopt.space', 'Integer'),
    ('skopt.space', 'Categorical'),
    ('skopt.utils', 'use_named_args')
]

successful_imports = []
failed_imports = []

for package, alias in packages:
    try:
        # Check if the package is already imported
        if alias:
            if alias in globals():
                print(f"Skipping already imported package: {package}")
                continue
            globals()[alias] = importlib.import_module(package)
        else:
            if package in sys.modules:
                print(f"Skipping already imported package: {package}")
                continue
            importlib.import_module(package)
        successful_imports.append(package)
        print(f"Successfully imported {package}")
    except Exception as e:
        failed_imports.append((package, str(e)))
        print(f"Failed to import {package}: {e}")

print("\nSummary:")
print("Successful imports:")
for package in successful_imports:
    print(f"- {package}")

print("\nFailed imports:")
for package, error in failed_imports:
    print(f"- {package}: {error}")

def plot_predictions(name, y_pred_train, y_train, y_pred_test, y_valtest, logscaled=True, xylim=True):

    xlims, ylims = 1e-5, 1e-5
    plt.figure(figsize=(8, 12), dpi=1200)
    gs = GridSpec(3, 2, height_ratios=[1, 2, 2], hspace=0.3, wspace=0.3)  # Adjusted height ratio and spacing

    # KDE plot for Train in the first row, first column
    ax1 = plt.subplot(gs[0, 0])  # ax1 is used for KDE (Train)
    sns.kdeplot(y_pred_train.squeeze(), color='green', label='Predicted Values')
    sns.kdeplot(y_train.squeeze(), color='royalblue', label='Actual Values')
    plt.xlabel('Value', fontsize=6,fontfamily='Liberation Sans')
    plt.ylabel('Distribution (%)', fontsize=6,fontfamily='Liberation Sans')
    plt.xticks(fontsize=6,fontfamily='Liberation Sans')
    plt.yticks(fontsize=6,fontfamily='Liberation Sans')
    ax1.set_xlim(left=0)
    ax1.annotate('a. KDE (Training)', xy=(0.5, -0.28), xycoords='axes fraction', 
                 ha="center", va="top", fontsize='medium', fontweight='semibold', fontstretch='semi-expanded',fontfamily='Liberation Sans')
    if logscaled:
        plt.xscale('log')
    if xylim:       
        plt.xlim([xlims, plt.xlim()[1]])
        plt.ylim([ylims, plt.ylim()[1]])
    plt.legend(fontsize=6)

    # KDE plot for Test in the first row, second column
    ax2 = plt.subplot(gs[0, 1])  # ax2 is used for KDE (Test)
    sns.kdeplot(y_pred_test.squeeze(), color='green', label='Predicted Values')
    sns.kdeplot(y_valtest.squeeze(), color='royalblue', label='Actual Values')
    plt.xlabel('Value', fontsize=6,fontfamily='Liberation Sans')
    plt.ylabel('Distribution (%)', fontsize=6,fontfamily='Liberation Sans')
    plt.xticks(fontsize=5,fontfamily='Liberation Sans')
    plt.yticks(fontsize=6,fontfamily='Liberation Sans')
    ax2.set_xlim(left=0)
    ax2.annotate('b. KDE (Testing)', xy=(0.5, -0.28), xycoords='axes fraction', 
                 ha="center", va="top", fontsize='medium', fontweight='semibold', fontstretch='semi-expanded',fontfamily='Liberation Sans')
    if logscaled:
        plt.xscale('log')
    if xylim:       
        plt.xlim([xlims, plt.xlim()[1]])
        plt.ylim([ylims, plt.ylim()[1]])
    plt.legend(fontsize=6)

    # Train scatterplot for the predicted vs actual values in the second row, first column
    ax3 = plt.subplot(gs[1, 0])  # ax3 is used for Scatter (Train)
    sns.scatterplot(x=y_pred_train.ravel(), y=y_train.squeeze(), color='green', s=15)
    plt.plot(y_train, y_train, color="black", linestyle="--", linewidth=1)
    ax3.annotate('c. Training', xy=(0.5, -0.14), xycoords='axes fraction', 
                 ha="center", va="top", fontsize='medium', fontweight='semibold', fontstretch='semi-expanded',fontfamily='Liberation Sans')
    plt.xlabel('Predicted', fontsize=6,fontfamily='Liberation Sans')
    plt.ylabel('Actual', fontsize=6,fontfamily='Liberation Sans')
    plt.xticks(fontsize=6,fontfamily='Liberation Sans')
    plt.yticks(fontsize=6,fontfamily='Liberation Sans')
    if logscaled:
        plt.xscale('log')
        plt.yscale('log')
    if xylim:       
        plt.xlim([xlims, plt.xlim()[1]])
        plt.ylim([ylims, plt.ylim()[1]])

    # Test scatterplot for the predicted vs actual values in the second row, second column
    ax4 = plt.subplot(gs[1, 1])  # ax4 is used for Scatter (Test)
    sns.scatterplot(x=y_pred_test.ravel(), y=y_valtest.squeeze(), color='royalblue', s=15)
    plt.plot(y_valtest, y_valtest, color="black", linestyle="--", linewidth=1)
    ax4.annotate('d. Testing', xy=(0.5, -0.14), xycoords='axes fraction', 
                 ha="center", va="top", fontsize='medium', fontweight='semibold', fontstretch='semi-expanded',fontfamily='Liberation Sans')
    plt.xlabel('Predicted', fontsize=6,fontfamily='Liberation Sans')
    plt.ylabel('Actual', fontsize=6,fontfamily='Liberation Sans')
    plt.xticks(fontsize=6,fontfamily='Liberation Sans')
    plt.yticks(fontsize=6,fontfamily='Liberation Sans')
    if logscaled:
        plt.xscale('log')
        plt.yscale('log')
    if xylim:       
        plt.xlim([xlims, plt.xlim()[1]])
        plt.ylim([ylims, plt.ylim()[1]])
        
    #plt.tight_layout()
    filename = f'TruePred_{name}.svg'
    plt.savefig(filename, bbox_inches='tight', format='svg')

class DynamicPINN(nn.Module):
    def __init__(self, layer_sizes, activations, dropout_rate=0.3):
        super(DynamicPINN, self).__init__()

        # Build the layers for each phase
        self.phase0_layers = self._build_layers(layer_sizes[0], activations[0], dropout_rate)
        self.phase1_layers = self._build_layers(layer_sizes[1], activations[1], dropout_rate)
        self.phase2_layers = self._build_layers(layer_sizes[2], activations[2], dropout_rate)

        # Ensure the output size of the last layer of phase 0 matches the input size of the first layer of phase 1
        if layer_sizes[0][-1] != layer_sizes[1][0]:
            raise ValueError("The output size of the last layer of phase 0 must match the input size of the first layer of phase 1.")
        
        # Ensure the output size of the last layer of phase 1 matches the input size of the first layer of phase 2
        if layer_sizes[1][-1] != layer_sizes[2][0]:
            raise ValueError("The output size of the last layer of phase 1 must match the input size of the first layer of phase 2.")
        
        # Final output layer
        self.final_layer = nn.Sequential(
            nn.Linear(layer_sizes[2][-1], 1),
            nn.ReLU()
        )
        
    def _build_layers(self, layer_sizes, activations, dropout_rate):
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            layers.append(self._get_activation(activations[i]))
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
        return nn.Sequential(*layers)

    def _get_activation(self, name):
        activations = {
            'relu': nn.ReLU(), 'linear': nn.Identity(), 'leaky_relu': nn.LeakyReLU(),
            'selu': nn.SELU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh(),
            'softplus': nn.Softplus(), 'softsign': nn.Softsign(), 'elu': nn.ELU(),
            'celu': nn.CELU(), 'prelu': nn.PReLU(), 'rrelu': nn.RReLU(), 'gelu': nn.GELU()
        }
        return activations.get(name, nn.Identity())

    def forward(self, x):
        x = self.phase0_layers(x)
        x = self.phase1_layers(x)
        x = self.phase2_layers(x)
        x = self.final_layer(x)
        return x

def train_dynamic_pinn(model, X_train, y_train, X_val, y_val, n_epochs_mse, n_epochs_pos, n_epochs_neg, feature_idx_pos, feature_idx_neg, positive_penalty_scale, negative_penalty_scale, lr=1e-3, optimizer_type="Adam"):
    optimizer = getattr(torch.optim, optimizer_type)(model.parameters(), lr=lr, weight_decay=1e-4)  # Add L2 regularization
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    train_loss, val_loss = [], []

    total_epochs = n_epochs_mse + n_epochs_pos + n_epochs_neg

    for epoch in range(total_epochs):
        if epoch < n_epochs_mse:
            loss_function = F.mse_loss
        elif epoch < n_epochs_mse + n_epochs_pos:
            loss_function = positive_correlation_loss
            feature_idx = feature_idx_pos
            penalty_scale = positive_penalty_scale
        else:
            loss_function = negative_correlation_loss
            feature_idx = feature_idx_neg
            penalty_scale = negative_penalty_scale

        model.train()
        optimizer.zero_grad()
        y_pred_train = model(X_train)
        
        if epoch < n_epochs_mse:
            loss_train = loss_function(y_pred_train, y_train)
        else:
            loss_train = loss_function(y_pred_train, y_train, X_train, feature_idx, penalty_scale)
        
        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Apply gradient clipping
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val)
            loss_val = F.mse_loss(y_pred_val, y_val)
        
        train_loss.append(loss_train.item())
        val_loss.append(loss_val.item())

        if epoch % 200 == 0 or epoch == total_epochs - 1:
            print(f"Epoch {epoch}: Train Loss = {loss_train.item():.4f}, Val Loss = {loss_val.item():.4f}")

    return train_loss, val_loss

def positive_correlation_loss(y_pred, y_true, X, feature_idx, penalty_scale):
    mse_loss = F.mse_loss(y_pred, y_true)
    target_feature = X[:, feature_idx]
    pred_mean = torch.mean(y_pred)
    feature_mean = torch.mean(target_feature)
    covariance = torch.mean((y_pred - pred_mean) * (target_feature - feature_mean))
    pred_std = torch.std(y_pred) + 1e-5  # Prevent division by zero
    feature_std = torch.std(target_feature) + 1e-5  # Prevent division by zero
    correlation = covariance / (pred_std * feature_std)
    penalty = 0 if correlation > 0 else penalty_scale * torch.abs(correlation)
    combined_loss = mse_loss + penalty
    return combined_loss

def negative_correlation_loss(y_pred, y_true, X, feature_idx, penalty_scale):
    mse_loss = F.mse_loss(y_pred, y_true)
    target_feature = X[:, feature_idx]
    pred_mean = torch.mean(y_pred)
    feature_mean = torch.mean(target_feature)
    covariance = torch.mean((y_pred - pred_mean) * (target_feature - feature_mean))
    pred_std = torch.std(y_pred) + 1e-5  # Prevent division by zero
    feature_std = torch.std(target_feature) + 1e-5  # Prevent division by zero
    correlation = covariance / (pred_std * feature_std)
    penalty = 0 if correlation < 0 else penalty_scale * torch.abs(correlation)
    combined_loss = mse_loss + penalty
    return combined_loss


def evaluate_model_with_kfold(trial):
    val_mse_list = []
    # K-Fold Cross-Validation
    k_folds = 3
    kf = KFold(n_splits=k_folds, random_state=101, shuffle=True)
    for train_index, val_index in kf.split(input_train_data):
        X_train, X_val = input_train_data[train_index], input_train_data[val_index]
        y_train, y_val = output_train_data[train_index], output_train_data[val_index]

        X_train = torch.tensor(X_train, dtype=torch.float32).cpu() if isinstance(X_train, np.ndarray) else X_train.clone().detach().cpu()
        y_train = torch.tensor(y_train, dtype=torch.float32).cpu() if isinstance(y_train, np.ndarray) else y_train.clone().detach().cpu()
        X_val = torch.tensor(X_val, dtype=torch.float32).cpu() if isinstance(X_val, np.ndarray) else X_val.clone().detach().cpu()
        y_val = torch.tensor(y_val, dtype=torch.float32).cpu() if isinstance(y_val, np.ndarray) else y_val.clone().detach().cpu()

        # Suggest hyperparameters using Optuna
        layer_sizes_phase0 = [X_train.shape[1]] + [trial.suggest_int(f'layer_size_phase0_{i}', 28, 112) for i in range(trial.suggest_int('num_layers_phase0', 2, 4))]
        layer_sizes_phase1 = [layer_sizes_phase0[-1]] + [trial.suggest_int(f'layer_size_phase1_{i}', 28, 112) for i in range(trial.suggest_int('num_layers_phase1', 2, 4))]
        layer_sizes_phase2 = [layer_sizes_phase1[-1]] + [trial.suggest_int(f'layer_size_phase2_{i}', 28, 112) for i in range(trial.suggest_int('num_layers_phase2', 2, 4))]
        layer_sizes = [layer_sizes_phase0, layer_sizes_phase1, layer_sizes_phase2]

        # Ensure proper layer size alignment between phases
        for i in range(2):
            if layer_sizes[i][-1] != layer_sizes[i + 1][0]:
                raise ValueError(f"The output size of the last layer of phase {i} must match the input size of the first layer of phase {i + 1}.")

        n_epochs_phase0 = trial.suggest_int('n_epochs_phase0', 300, 600, step=100)
        n_epochs_phase1 = trial.suggest_int('n_epochs_phase1', 300, 600, step=100)
        n_epochs_phase2 = trial.suggest_int('n_epochs_phase2', 300, 600, step=100)

        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        activation_functions = ['relu', 'softplus', 'gelu', 'selu', 'sigmoid']
        activations_phase0 = [trial.suggest_categorical(f'activation_phase0_{i}', activation_functions) for i in range(len(layer_sizes_phase0) - 1)]
        activations_phase1 = [trial.suggest_categorical(f'activation_phase1_{i}', activation_functions) for i in range(len(layer_sizes_phase1) - 1)]
        activations_phase2 = [trial.suggest_categorical(f'activation_phase2_{i}', activation_functions) for i in range(len(layer_sizes_phase2) - 1)]
        activations = [activations_phase0, activations_phase1, activations_phase2]

        optimizer_type = trial.suggest_categorical('optimizer_type', ['Adam', 'SGD'])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3, step=0.05)
        positive_penalty_scale = trial.suggest_float('positive_penalty_scale', 0.0, 5.0, step=0.5)
        negative_penalty_scale = trial.suggest_float('negative_penalty_scale', 0.0, 5.0, step=0.5)

        # Feature indices and desired signs for the custom loss functions
        target_feature_indices = [17, 11]  # ESAL (17), LyrC (11)
        desired_signs = [1, -1]  # Positive for ESAL, Negative for LyrC

        # Initialize and train the model
        model = DynamicPINN(
            layer_sizes=layer_sizes,
            activations=activations,
            dropout_rate=dropout_rate
        )

        train_dynamic_pinn(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            n_epochs_mse=n_epochs_phase0,
            n_epochs_pos=n_epochs_phase1,
            n_epochs_neg=n_epochs_phase2,
            feature_idx_pos=target_feature_indices[0],
            feature_idx_neg=target_feature_indices[1],
            positive_penalty_scale=positive_penalty_scale,
            negative_penalty_scale=negative_penalty_scale,
            lr=lr,
            optimizer_type=optimizer_type
        )

        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val)
        val_mse = torch.nn.functional.mse_loss(y_pred_val, y_val).item()
        val_mse_list.append(val_mse)

    avg_val_mse = np.mean(val_mse_list)
    return avg_val_mse

# Optuna optimization
study = optuna.create_study(direction='minimize', study_name='NNPINNs01')
study.optimize(evaluate_model_with_kfold, n_trials=35)

best_params = study.best_params
best_score = study.best_value

print(f'Best Score: {best_score:.4f}')
print('Best hyperparameters: ', best_params)
