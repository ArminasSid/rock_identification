import os
import pandas as pd
import random
import json
import torch
import torchvision

from detecto.config import config
from detecto.utils import default_transforms, filter_top_predictions, xml_to_csv, _is_iterable, read_image
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from detecto.core import DataLoader, Dataset

class Model:

    DEFAULT = 'fasterrcnn_resnet50_fpn'
    MOBILENET = 'fasterrcnn_mobilenet_v3_large_fpn'
    MOBILENET_320 = 'fasterrcnn_mobilenet_v3_large_320_fpn'

    def __init__(self, classes=None, device=None, pretrained=True, model_name=DEFAULT):
        self._device = device if device else config['default_device']

        # Load a model pre-trained on COCO
        if model_name == self.DEFAULT:
            self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
        elif model_name == self.MOBILENET:
            self._model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=pretrained)
        elif model_name == self.MOBILENET_320:
            self._model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=pretrained)
        else:
            raise ValueError(f'Invalid value {model_name} for model_name. ' +
                             f'Please choose between {self.DEFAULT}, {self.MOBILENET}, and {self.MOBILENET_320}.')

        if classes:
            # Get the number of input features for the classifier
            in_features = self._model.roi_heads.box_predictor.cls_score.in_features
            # Replace the pre-trained head with a new one (note: +1 because of the __background__ class)
            self._model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes) + 1)
            self._disable_normalize = False
        else:
            classes = config['default_classes']
            self._disable_normalize = True

        self._model.to(self._device)

        # Mappings to convert from string labels to ints and vice versa
        self._classes = ['__background__'] + classes
        self._int_mapping = {label: index for index, label in enumerate(self._classes)}

    # Returns the raw predictions from feeding an image or list of images into the model
    def _get_raw_predictions(self, images):
        self._model.eval()

        with torch.no_grad():
            # Convert image into a list of length 1 if not already a list
            if not _is_iterable(images):
                images = [images]

            # Convert to tensor and normalize if not already
            if not isinstance(images[0], torch.Tensor):
                if self._disable_normalize:
                    defaults = transforms.Compose([transforms.ToTensor()])
                else:
                    defaults = default_transforms()
                images = [defaults(img) for img in images]

            # Send images to the specified device
            images = [img.to(self._device) for img in images]

            preds = self._model(images)
            # Send predictions to CPU if not already
            preds = [{k: v.to(torch.device('cpu')) for k, v in p.items()} for p in preds]
            return preds

    def predict(self, images):
        # Convert all to lists but keep track if a single image was given
        is_single_image = not _is_iterable(images)
        images = [images] if is_single_image else images
        preds = self._get_raw_predictions(images)

        results = []
        for pred in preds:
            # Convert predicted ints into their corresponding string labels
            result = ([self._classes[val] for val in pred['labels']], pred['boxes'], pred['scores'])
            results.append(result)

        return results[0] if is_single_image else results


    def predict_top(self, images):
        predictions = self.predict(images)

        # If tuple but not list, then images is a single image
        if not isinstance(predictions, list):
            return filter_top_predictions(*predictions)

        results = []
        for pred in predictions:
            results.append(filter_top_predictions(*pred))

        return results

    def fit(self, dataset, val_dataset=None, epochs=10, learning_rate=0.005, momentum=0.9,
            weight_decay=0.0005, gamma=0.1, lr_step_size=3, verbose=True):

        if verbose and self._device == torch.device('cpu'):
            print('It looks like you\'re training your model on a CPU. '
                  'Consider switching to a GPU; otherwise, this method '
                  'can take hours upon hours or even days to finish. '
                  'For more information, see https://detecto.readthedocs.io/'
                  'en/latest/usage/quickstart.html#technical-requirements')

        # If doing custom training, the given images will most likely be
        # normalized. This should fix the issue of poor performance on
        # default classes when normalizing, so resume normalizing. TODO
        if epochs > 0:
            self._disable_normalize = False

        # Convert dataset to data loader if not already
        if not isinstance(dataset, DataLoader):
            dataset = DataLoader(dataset, shuffle=True)

        if val_dataset is not None and not isinstance(val_dataset, DataLoader):
            val_dataset = DataLoader(val_dataset)

        losses = {
            'training': [],
            'validation': []
        }
        # Get parameters that have grad turned on (i.e. parameters that should be trained)
        parameters = [p for p in self._model.parameters() if p.requires_grad]
        # Create an optimizer that uses SGD (stochastic gradient descent) to train the parameters
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        # Create a learning rate scheduler that decreases learning rate by gamma every lr_step_size epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

        # Train on the entire dataset for the specified number of times (epochs)
        for epoch in range(epochs):
            if verbose:
                print('Epoch {} of {}'.format(epoch + 1, epochs))

            # Training step
            self._model.train()

            if verbose:
                print('Begin iterating over training dataset')

            iterable = tqdm(dataset, position=0, leave=True) if verbose else dataset
            training_avg_loss = 0
            for images, targets in iterable:
                self._convert_to_int_labels(targets)
                images, targets = self._to_device(images, targets)

                # Calculate the model's loss (i.e. how well it does on the current
                # image and target, with a lower loss being better)
                loss_dict = self._model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())
                training_avg_loss += total_loss.item()

                # Zero any old/existing gradients on the model's parameters
                optimizer.zero_grad()
                # Compute gradients for each parameter based on the current loss calculation
                total_loss.backward()
                # Update model parameters from gradients: param -= learning_rate * param.grad
                optimizer.step()
            
            training_avg_loss /= len(dataset.dataset)
            losses['training'].append(training_avg_loss)
            if verbose:
                print(f'Training loss: {training_avg_loss}')

            # Validation step
            if val_dataset is not None:
                validation_avg_loss = 0
                with torch.no_grad():
                    if verbose:
                        print('Begin iterating over validation dataset')

                    iterable = tqdm(val_dataset, position=0, leave=True) if verbose else val_dataset
                    for images, targets in iterable:
                        self._convert_to_int_labels(targets)
                        images, targets = self._to_device(images, targets)
                        loss_dict = self._model(images, targets)
                        total_loss = sum(loss for loss in loss_dict.values())
                        validation_avg_loss += total_loss.item()

                validation_avg_loss /= len(val_dataset.dataset)
                losses['validation'].append(validation_avg_loss)

                if verbose:
                    print('Validation loss: {}'.format(validation_avg_loss))

            # Update the learning rate every few epochs
            lr_scheduler.step()

        if len(losses) > 0:
            return losses

    def get_internal_model(self):
        return self._model

    def save(self, file):
        torch.save(self._model.state_dict(), file)

    @staticmethod
    def load(file, classes):
        model = Model(classes)
        model._model.load_state_dict(torch.load(file, map_location=model._device))
        return model

    # Converts all string labels in a list of target dicts to
    # their corresponding int mappings
    def _convert_to_int_labels(self, targets):
        for idx, target in enumerate(targets):
            # get all string labels for objects in a single image
            labels_array = target['labels']
            # convert string labels into one hot encoding
            labels_int_array = [self._int_mapping[class_name] for class_name in labels_array]
            target['labels'] = torch.tensor(labels_int_array)

    # Sends all images and targets to the same device as the model
    def _to_device(self, images, targets):
        images = [image.to(self._device) for image in images]
        targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]
        return images, targets


def write_losses_to_file(losses: dict, file: str):
    with open(file, 'w') as fp:
        json.dump(losses, fp)


def main():
    print(f'Cuda is available - {torch.cuda.is_available()}')
    output_file = 'model_b2'
    dataset = 'dataset_b2'

    model_output_path = f'{output_file}.pth'
    stats_output_path = f'{output_file}.json'

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    training_dataset = Dataset(f'{dataset}/training_dataset/images', transform=transform)
    validation_dataset = Dataset(f'{dataset}/validation_dataset/images')
    training_loader = DataLoader(dataset=training_dataset, batch_size=2, num_workers=2, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=2, num_workers=2, shuffle=False)

    model = Model(classes=['rock', 'rock_underwater'], pretrained=True, model_name=Model.DEFAULT)

    # Training
    losses = model.fit(
        dataset=training_loader,
        val_dataset=validation_loader,
        verbose=True,
        epochs=20
    )

    model.save(file=model_output_path)
    write_losses_to_file(losses=losses, file=stats_output_path)

if __name__=='__main__':
    main()
