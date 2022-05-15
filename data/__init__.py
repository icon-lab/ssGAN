import torch.utils.data
from data.base_data_loader import BaseDataLoader
import numpy as np, h5py
import random
def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode !='aligned_mat' and opt.dataset_mode !='unaligned_mat':
        if opt.dataset_mode == 'aligned':
            from data.aligned_dataset import AlignedDataset
            dataset = AlignedDataset()
        elif opt.dataset_mode == 'unaligned':
            from data.unaligned_dataset import UnalignedDataset
            dataset = UnalignedDataset()
        elif opt.dataset_mode == 'single':
            from data.single_dataset import SingleDataset
            dataset = SingleDataset()
        else:
            raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)  
            print("dataset [%s] was created" % (dataset.name()))
        dataset.initialize(opt)
         
    #custom data loader    
    if opt.dataset_mode == 'aligned_mat' or opt.dataset_mode == 'unaligned_mat':   
        #data location
        if opt.phase == "train":
            temp_dataset_name = "Train"
        if opt.phase == "test":
            temp_dataset_name = "Test"
        if opt.phase == "val":
            temp_dataset_name = "Val"
            
            
            samp_traj_val_T2_9.mat
            images_val_T1.mat
            
            
        source_image_file = opt.dataroot + "/images_" + opt.phase + "_" + opt.source_contrast + ".mat"
        target_image_file = opt.dataroot + "/images_" + opt.phase + "_" + opt.target_contrast + ".mat"
        source_samp_traj_file = opt.dataroot + "/samp_traj_" + opt.phase + "_" + opt.source_contrast + "_" + str(opt.acc_ratio_source) + ".mat" 
        target_samp_traj_file = opt.dataroot + "/samp_traj_" + opt.phase + "_" + opt.target_contrast + "_" + str(opt.acc_ratio_target) + ".mat" 
        
        f = h5py.File(source_image_file,'r') 
        source_image_arr = np.array(f['image_arr'])
        
        f = h5py.File(target_image_file,'r') 
        target_image_arr = np.array(f['image_arr'])
        
        f = h5py.File(source_samp_traj_file,'r') 
        source_traj_arr = np.array(f['traj_arr'])
        
        f = h5py.File(target_samp_traj_file,'r') 
        target_traj_arr = np.array(f['traj_arr'])
        
        if opt.phase == "train":
            cross_sections_train=np.array([76,172,263,354,425,516,607,688,779,875,971,1057,1148,1224,1305,1406,1492,1583,1674,1755,1851,1942,2013,2099,2185,2276,2357,2443,2544,2640,2731,2807,2893,2969,3055,3126,3212,3293,3369,3460,3531,3607,3688,3774,3845,3926,4007,4088,4169,4260,4341,4437,4513,4609,4700,4791,4877,4968,5049,5135,5216,5297,5383,5459]);
            sample_indices = np.arange(cross_sections_train[opt.num_total_subjects-1])
            np.random.shuffle(sample_indices)
            source_image_arr = source_image_arr[sample_indices,:,:]
            target_image_arr = target_image_arr[sample_indices,:,:]
            source_traj_arr = source_traj_arr[sample_indices,:,:]
            target_traj_arr = target_traj_arr[sample_indices,:,:]
        
        data_source_image_arr = np.expand_dims(np.array(source_image_arr),axis=0)
        data_target_image_arr = np.expand_dims(np.array(target_image_arr),axis=0)
        data_source_traj_arr = np.expand_dims(np.array(source_traj_arr),axis=0)
        data_target_traj_arr = np.expand_dims(np.array(target_traj_arr),axis=0)
        
        data_x = np.concatenate((data_source_image_arr,data_source_traj_arr,np.zeros(data_source_traj_arr.shape))).astype(np.float32)
        data_y = np.concatenate((data_target_image_arr,data_target_traj_arr,np.zeros(data_target_traj_arr.shape))).astype(np.float32)
        data_x[data_x<0] = 0
        data_y[data_y<0] = 0   
        print(np.unique(data_x[1,:,:,:]))
        dataset=[]
        for train_sample in range(data_x.shape[1]):
            data_x[:,train_sample,:,:]=(data_x[:,train_sample,:,:]-0.5)/0.5
            data_y[:,train_sample,:,:]=(data_y[:,train_sample,:,:]-0.5)/0.5
            dataset.append({'A': torch.from_numpy(data_x[:,train_sample,:,:]), 'B':torch.from_numpy(data_y[:,train_sample,:,:]), 
            'A_paths':opt.dataroot, 'B_paths':opt.dataroot})
        print('#training images = %d' % train_sample)
        print(data_x.shape)
        print(data_y.shape)        
    #else:
    #    raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)
    return dataset 



class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
