import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import numpy as np, h5py 
from skimage.measure import compare_psnr as psnr
if __name__ == '__main__':
    opt = TrainOptions().parse()
    #Training data
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    #validation data
    opt.phase='test'
    data_loader_val = CreateDataLoader(opt)
    dataset_val = data_loader_val.load_data()
    dataset_size_val = len(data_loader_val)
    print('#Validation images = %d' % dataset_size)
    if opt.model=='cycle_gan':
        L1_avg=np.zeros([2,opt.niter + opt.niter_decay,len(dataset_val)])      
        psnr_avg=np.zeros([2,opt.niter + opt.niter_decay,len(dataset_val)])            
    else:
        L1_avg=np.zeros([opt.niter + opt.niter_decay,len(dataset_val)])      
        psnr_avg=np.zeros([opt.niter + opt.niter_decay,len(dataset_val)])       
    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        #Training step
        opt.phase='train'
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                if opt.dataset_mode=='aligned_mat':
                    temp_visuals=model.get_current_visuals()
                    #temp_visuals['real_A']=temp_visuals['real_A'][:,:,0:3]
                    #temp_visuals['real_B']=temp_visuals['real_B'][:,:,0:3]
                    #temp_visuals['fake_B']=temp_visuals['fake_B'][:,:,0:3]
                    visualizer.display_current_results(temp_visuals, epoch, save_result)
                elif  opt.dataset_mode=='unaligned_mat':   
                    temp_visuals=model.get_current_visuals()
                    temp_visuals['real_A']=temp_visuals['real_A'][:,:,0:3]
                    temp_visuals['real_B']=temp_visuals['real_B'][:,:,0:3]
                    temp_visuals['fake_A']=temp_visuals['fake_A'][:,:,0:3]
                    temp_visuals['fake_B']=temp_visuals['fake_B'][:,:,0:3]
                    temp_visuals['rec_A']=temp_visuals['rec_A'][:,:,0:3]
                    temp_visuals['rec_B']=temp_visuals['rec_B'][:,:,0:3]
                    if opt.lambda_identity>0:
                      temp_visuals['idt_A']=temp_visuals['idt_A'][:,:,0:3]
                      temp_visuals['idt_B']=temp_visuals['idt_B'][:,:,0:3]                    
                    visualizer.display_current_results(temp_visuals, epoch, save_result)                    
                else:
                    temp_visuals=model.get_current_visuals()
                    #temp_visuals['real_A']=np.concatenate((temp_visuals['real_A'][:,:,0:2],np.zeros((256,256,1))),axis=2)
                    #temp_visuals['real_A']=temp_visuals['real_A'][:,:,0:3]
                    #temp_visuals['real_B']=temp_visuals['real_B'][:,:,0:3]
                    #temp_visuals['fake_B']=temp_visuals['fake_B'][:,:,0:3]
                    visualizer.display_current_results(temp_visuals, epoch, save_result)                    
                    

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

            iter_data_time = time.time()
        #Validaiton step   
        print(opt.dataset_mode)
        opt.phase='val'
        for i, data_val in enumerate(dataset_val):
            model.set_input(data_val)
            model.test()
            if opt.model=='cycle_gan':
                fake_im_B=model.fake_B.cpu().numpy()
                fake_im_A=model.fake_A.cpu().numpy()
                real_im_A=model.real_A.cpu().data.numpy() 
                real_im_B=model.real_B.cpu().data.numpy() 
                if (opt.dataset_mode=='unaligned_mat' or opt.dataset_mode=='unaligned'):# and opt.output_nc==1:
                    slice_sel=0
                else:
                    slice_sel=1
                fake_im_A=fake_im_A[0,slice_sel,:,:]*0.5+0.5
                fake_im_B=fake_im_B[0,slice_sel,:,:]*0.5+0.5
                real_im_A=real_im_A[0,slice_sel,:,:]*0.5+0.5
                real_im_B=real_im_B[0,slice_sel,:,:]*0.5+0.5
                #L1_avg[0,epoch-1,i]=abs(fake_im_A-real_im_B).mean()
                #psnr_avg[0,epoch-1,i]=psnr(fake_im_A/fake_im_A.max(),real_im_A/real_im_A.max())
                #L1_avg[1,epoch-1,i]=abs(fake_im_B-real_im_A).mean()
                #psnr_avg[1,epoch-1,i]=psnr(fake_im_B/fake_im_B.max(),real_im_B/real_im_B.max())
            else:    
                fake_im=model.fake_B.cpu().data.numpy()
                real_im=model.real_B.cpu().data.numpy() 
                real_im=real_im*0.5+0.5
                fake_im=fake_im*0.5+0.5
                #L1_avg[epoch-1,i]=abs(fake_im-real_im).mean()
                #psnr_avg[epoch-1,i]=psnr(fake_im/fake_im.max(),real_im/real_im.max())
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
        #print(psnr_avg[epoch-1,:].mean())   
        #print(L1_avg[epoch-1,:].mean()) 
        #f = h5py.File('/auto/data/myurt/GATED_FUSION_VAL/'+opt.name+'.mat',  "w")
        #f.create_dataset('L1_avg', data=L1_avg)
        #f.create_dataset('psnr_avg', data=psnr_avg)
        #f.close()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
