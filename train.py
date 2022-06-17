import torch
from detecto.core import Model, Dataset, DataLoader


def main():
    print(f'Cuda is available - {torch.cuda.is_available()}')

    model_output_path = 'model.pth'

    training_dataset = Dataset('training_dataset/images')
    validation_dataset = Dataset('validation_dataset/images')
    training_loader = DataLoader(dataset=training_dataset, batch_size=4, num_workers=8, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=4, num_workers=8, shuffle=False)

    model = Model(classes=['rock'], pretrained=True, model_name=Model.DEFAULT)

    # Training
    losses = model.fit(
        dataset=training_loader,
        val_dataset=validation_loader,
        verbose=True,
        epochs=100
    )

    model.save(file=model_output_path)


if __name__=='__main__':
    main()
