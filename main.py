import os
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import torch
import config
import unet


#torch.cuda.init() ########## REMOVE
#torch.autograd.set_detect_anomaly(True) ######
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ########
plt.style.use('ggplot')

# Get model info
model_type, model_version = config.get_modelinfo()
# Load specified hyperparameters in config.py
hyperparameters = config.get_hyperparameters(model_type)
# Load/create paths (wd/results/[plots, json, models]) and training/validation image label files (.csv)
train_lab_file, valid_lab_file = config.get_paths()
# Extract training & validation images as lists
train_data_list_aug, valid_data_list, dataset_list = utils.get_datalists(train_lab_file, valid_lab_file)
# Calculate normalization from complete image dataset
############################################################################################################################################################  REMOVE!!!!!!
norm_mean = [0.53778025, 0.55671272, 0.5727046]
norm_std = [0.11824406, 0.11022782, 0.0986911]
#norm_mean, norm_std = utils.calculate_normalization(dataset_list)

# Load transformations of training and validation images, including normalization
train_transforms, valid_transforms = config.get_transforms(norm_mean, norm_std, hyperparameters['image_size'])

# Load training dataset with custom CloudDataset class, apply defined transformations
train_set = CloudDataset(
    img_lab_list=train_data_list_aug,
    label_category=hyperparameters['label_category'],
    transform=train_transforms
)

# Load validation dataset with custom CloudDataset class, apply defined transformations
valid_set = CloudDataset(
    img_lab_list=valid_data_list,
    label_category=hyperparameters['label_category'],
    transform=valid_transforms
)

# Create final training and validation dataset with specified batch size
train_loader, valid_loader = utils.data_loader(hyperparameters, train_set, valid_set)

if model_type == 'Ensemble':
    ensemble_models_list = []
    ensemble_models_layer_structure = []
    ensemble_models_dict, decision_mode = config.get_model_ensemble_info()
    for e_model_type, (e_model_version, _) in ensemble_models_dict.items():
        model, model_name = CNN_models.load_model(e_model_type, e_model_version, hyperparameters)
        # Update model with custom modifications (layer freezing, changed fc/classifier layer(s))
        model, layer_structure = config.get_model_structure(model, e_model_type, hyperparameters)
        ensemble_models_list.append(model)
        ensemble_models_layer_structure.append(layer_structure)

    model_list = [f'{key}{value1}' for key, (value1, _) in ensemble_models_dict.items()]
    model_name = ', '.join(model_list)
    if len(model_list) < 5:
        model_name = f'Ensemble ({model_name})'
    else:
        model_name = f'Ensemble of {len(model_list)} models'
    if decision_mode == 'weighting':
        model_weights = [weight for _, weight in ensemble_models_dict.values()]
        model = CNN_models.EnsembleModel(models_list=ensemble_models_list, decision_mode=decision_mode,
                                         weights=model_weights)

    elif decision_mode == 'majority_voting' or 'averaging':
        model = CNN_models.EnsembleModel(models_list=ensemble_models_list, decision_mode=decision_mode,
                                         weights=None)
    else:
        raise ValueError(f'Wrong decision mode <{decision_mode}> for model ensemble prediction, check config.py')
    layer_structure = ensemble_models_layer_structure

else:
    # Load specified model (custom or pretrained)
    model, model_name = CNN_models.load_model(model_type, model_version, hyperparameters)
    # Update model with custom modifications (layer freezing, changed fc/classifier layer(s))
    model, layer_structure = config.get_model_structure(model, model_type, hyperparameters)

model.to(device)
# Get model optimizer, loss function and learning rate scheduler
optimizer, criterion, scheduler = config.get_model_params(model, hyperparameters)
# Update hyperparameters with optimizer, criterion & scheduler
hyperparameters = config.update_hyperparameters(hyperparameters, optimizer, criterion, scheduler)
# Load hyperparameters for plotting function
hyperparameters_legend = config.get_legend_hyperparameters(hyperparameters)

# Print model info
total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"MODEL SPECS:")
print(f"{total_params:,} total model parameters.")
print(f"{total_trainable_params:,} parameters unfreezed.")
print(f'')
print(f"{len(train_data_list_aug):,} total training images including augmentations")
print(f"{len(valid_data_list):,} total validation images")
print(f"{len(train_data_list_aug) + len(valid_data_list):,} total images")


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # Lists to keep track of losses and accuracies.
    start_time = time.time()
    datetime_str = datetime.now().strftime("%d-%m-%Y_%H-%M")

    early_stopped = False
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    lr_list = []
    best_valid_acc = 0
    print(device)
    early_stopper = EarlyStopping(patience=hyperparameters['es_patience'], min_delta=hyperparameters['es_mindelta'])
    # Start the training.
    for epoch in range(hyperparameters['epochs']):
        print(f"[INFO]: Epoch {epoch + 1} of {hyperparameters['epochs']}")
        train_epoch_loss, train_epoch_acc = train(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        valid_epoch_loss, valid_epoch_acc = validate(
            model,
            valid_loader,
            criterion,
            device
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

        if valid_epoch_acc > best_valid_acc:
            best_valid_acc = valid_epoch_acc
            print(f"\nBest validation accuracy: {round(best_valid_acc, 2)}")
            print(f"Saving best model for epoch: {epoch + 1}\n")
            model_save_name = f'{model_name}_{datetime_str}'
            torch.save(model.state_dict(), f'results/models/{model_save_name}.pth')

        if scheduler:
            if hyperparameters['scheduler_type'] == 'StepLR':
                scheduler.step()
            elif hyperparameters['scheduler_type'] == 'ReduceLROnPlateau':
                scheduler.step(valid_epoch_loss)
            else:
                raise ValueError(f"Wrong Scheduler '{hyperparameters['scheduler_type']}' defined")

            last_lr = format(scheduler._last_lr[0], ".2g")
            print(f'Learning rate: {last_lr}')
            if last_lr not in lr_list:
                lr_list.append(((epoch + 1), last_lr))
        else:
            print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
            lr_list = []

        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        # print(f"{} epochs until possible early stopping")
        print('-' * 50)
        if early_stopper.early_stop(valid_epoch_loss):
            early_stopped = True
            print(f'Early stopping triggered!')
            break

    end_time = time.time()
    runtime = str(timedelta(seconds=int(end_time - start_time)))

    # Save the loss and accuracy plots.
    save_stats(
        model,
        model_type,
        model_name,
        valid_loader,
        train_acc,
        valid_acc,
        train_loss,
        valid_loss,
        datetime_str,
        runtime,
        hyperparameters,
        lr_list,
        early_stopped,
        layer_structure,
        device,
    )

    create_plot(
        model,
        model_type,
        model_name,
        valid_loader,
        train_acc,
        valid_acc,
        train_loss,
        valid_loss,
        datetime_str,
        runtime,
        hyperparameters,
        hyperparameters_legend,
        lr_list,
        early_stopped,
        device,
        save_plt=True,
        load_from_json=False
    )

print('TRAINING COMPLETE')
