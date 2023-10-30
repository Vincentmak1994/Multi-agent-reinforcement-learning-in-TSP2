import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader 
from torch.autograd import Variable

class TSPDataset(Dataset):
    def __init__(self, data_size, seq_len):
        self.data_size = data_size
        self.seq_len = seq_len
        self.data = self._generate_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data['Points_List'][idx]).float()
        sample = {'Points':tensor}
        return sample
    
    def _generate_data(self):
        points_list = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description('Data points %i/%i' %(i+1, self.data_size))
            points_list.append(np.random.randint(30, size=(self.seq_len, 2)))
        
        return {'Points_List': points_list}
    
    def _to1hotvec(self, points):
        vec = np.zeros((len(points), self.seq_len))
        for i, v in enumerate(vec):
            v[points[i]] = 1
        return vec

def main():
    db = TSPDataset(10,1)
    print(db.data)
    train_loader = DataLoader(db, batch_size=10, shuffle=True, num_workers=1)
    print(train_loader)
    iterator = tqdm(train_loader, unit='Batch')
    for batch_id, sample_batch in enumerate(iterator):
        train_batch = Variable(sample_batch['Points'])
        print(train_batch)
        print(train_batch.transpose(1,2))

if __name__ == '__main__':
    main()




