from detecto.core import Model, Dataset


def main():
    training_dataset = Dataset('training_dataset/images')
    validation_dataset = Dataset('validation_dataset/images')

    model = Model(
        classes=['rock'],
        pretrained=True
    )
    # Training
    model.fit(
        dataset=training_dataset,
        val_dataset=validation_dataset,
        epochs=10,    
    )

    model.save('model.pth')


if __name__=='__main__':
    main()
