from utils import MenuDataset

menu_dataset  = MenuDataset("menu_price_1125.csv", "menu_vocab.p")
dataloader = DataLoader(menu_dataset, batch_size=4, shuffle=True)

# for i in range(len(menu_dataset)):
#
#     sample = menu_dataset[i]
#
#     print (i, sample["factor"], sample["words"], sample["price"])

for i_batch, sample_batched in enumerate(dataloader):
    print (i_batch, sample_batched["factor"], sample_batched["words"], sample_batched["price"])
