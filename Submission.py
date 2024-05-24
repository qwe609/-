import torch
import tqdm
from torch import nn
import pandas as pd


def submission(csv_path, test_loader, device, model):
    result_list = []
    model = model.to(device)
    test_loader = tqdm.tqdm(test_loader)
    with torch.no_grad():  # network does not update gradient during evaluation
        for i, data in enumerate(test_loader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            softmax_func = nn.Softmax(dim=1)  # dim=1 means the sum of rows is 1 按行softmax
            soft_output = softmax_func(outputs)  # soft_output is become two probability value
            predicted = soft_output.argmax(dim=1)  # the probability of dog
            for j in range(len(predicted)):
                result_list.append({
                    'id': labels[j].item(),
                    'label': predicted[j].item()
                     })

    # convert list to dataframe, and then generate csv format file
    columns = result_list[0].keys()
    result_list = {col: [anno[col] for anno in result_list] for col in columns}
    result_df = pd.DataFrame(result_list)
    result_df = result_df.sort_values("id")
    result_df.to_csv(csv_path, index=None)