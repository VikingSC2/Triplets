import os
import random
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt


class CelebaTripletDataSet(Dataset):
    def __init__(self, image_folder: str, identity_file: str, transform=None):

        self.image_folder = image_folder
        self.transform = transform
        self.person_to_images = {}

        # Загружаем соответствие "имя файла -> ID персоны"
        with open(identity_file, "r") as f:
            for line in f:
                image_name, person_id = line.strip().split()
                person_id = int(person_id)

                if person_id not in self.person_to_images:
                    self.person_to_images[person_id] = []

                # Меняем расширение .jpg → .png
                image_name = image_name.replace(".jpg", ".png")

                self.person_to_images[person_id].append(image_name)

        self.person_ids = list(self.person_to_images.keys())

    def __len__(self):
        """Возвращает количество уникальных персон в датасете"""
        return len(self.person_ids)

    def __getitem__(self, index):
        """Генерирует один триплет: Anchor, Positive, Negative"""
        person_id = self.person_ids[index]
        images = self.person_to_images[person_id]

        if len(images) < 2:
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        base_anchor, pos_anchor = random.sample(images, 2)
        negative_person_id = random.choice([pid for pid in self.person_ids if pid != person_id])
        neg_anchor = random.choice(self.person_to_images[negative_person_id])

        # Генерируем пути к файлам
        base_anch_path = os.path.join(self.image_folder, base_anchor)
        pos_anch_path = os.path.join(self.image_folder, pos_anchor)
        neg_anch_path = os.path.join(self.image_folder, neg_anchor)

        # Проверяем, что файлы существуют
        for path in [base_anch_path, pos_anch_path, neg_anch_path]:
            if not os.path.exists(path):
                print(f"Файл отсутствует: {path}")
                return self.__getitem__(random.randint(0, self.__len__() - 1))

        # Загружаем изображения
        base_anch = Image.open(base_anch_path).convert("RGB")
        pos_anch = Image.open(pos_anch_path).convert("RGB")
        neg_anch = Image.open(neg_anch_path).convert("RGB")

        if self.transform:
            base_anch = self.transform(base_anch)
            pos_anch = self.transform(pos_anch)
            neg_anch = self.transform(neg_anch)

        return {
            "person_id": person_id,
            "base_anchor": base_anch,
            "pos_anchor": pos_anch,
            "neg_anchor": neg_anch,
            "base_path": base_anch_path,
            "pos_path": pos_anch_path,
            "neg_path": neg_anch_path
        }

# Параметры датасета
dataset = CelebaTripletDataSet(
    image_folder="../img_align_celeba_png/img_align_celeba_png_unpacked/img_align_celeba_png",
    identity_file="identity_CelebA.txt",
)

# Загружаем один триплет
sample = dataset[1]

# Вывод информации
print(f"ID персоны: {sample['person_id']}")
print(f"Anchor: {sample['base_path']}")
print(f"Positive: {sample['pos_path']}")
print(f"Negative: {sample['neg_path']}")

# Визуализация триплета
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(Image.open(sample["base_path"]))
axes[0].set_title(f"Anchor\nID: {sample['person_id']}")
axes[0].axis("off")

axes[1].imshow(Image.open(sample["pos_path"]))
axes[1].set_title(f"Positive\nID: {sample['person_id']}")
axes[1].axis("off")

axes[2].imshow(Image.open(sample["neg_path"]))
axes[2].set_title(f"Negative\nID: другой человек")
axes[2].axis("off")

plt.show()

