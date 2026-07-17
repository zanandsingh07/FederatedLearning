from src import dataset

# Load dataset
df, encoder = dataset.load_dataset()

# Split dataset
train_df, valid_df, test_df = dataset.split_dataset(df)

# Create clients
client_datasets = dataset.create_iid_clients(train_df)

print("Total Clients :", len(client_datasets))

for i in range(3):
    print(f"\nClient {i+1}")
    print(client_datasets[i].shape)
    print(client_datasets[i]["class_name"].value_counts())