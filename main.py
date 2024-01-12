import models


def main():
    file_path = 'HIV_dataset.json'
    dataset = models.load_dataset(file_path)
    models.heart(dataset)

if __name__ == "__main__":
    main()