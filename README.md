# ssGAN
Official Pytorch Implementation of Semi-Supervised Learning of Mutually Accelerated MRI Synthesis without Fully-Sampled Ground Truths described in the [following](https://arxiv.org/abs/2011.14347) paper:

Mahmut Yurt, Salman Ul Hassan Dar, Muzaffer Özbey, Berk Tınaz, Kader Karlı Oğuz, Tolga Çukur Semi-Supervised Learning of Mutually Accelerated MRI Synthesis without Fully-Sampled Ground Truths. arXiv. 2022.

# Demo

Train
python train.py --gpu_ids 0 --dataroot [enter dataroot here] --name [enter name here] --source_contrast [enter source contrast here] --target_contrast [enter target contrast here] --model ssGAN --which_model_netG resnet_9blocks --dataset_mode aligned_mat --norm batch --niter 50 --niter_decay 50 --save_epoch_freq 25 --lambda_A 100 --checkpoints_dir [enter checkpoints directory here]

Test
python test.py --gpu_ids 0 --dataroot [enter dataroot here] --name [enter name here] --source_contrast [enter source contrast here] --target_contrast [enter target contrast here] --model ssGAN --which_model_netG resnet_9blocks --dataset_mode aligned_mat --norm batch --phase test --how_many 10000 --serial_batches --results_dir [enter results directory here] --checkpoints_dir [enter checkpoints directory here]

# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
@article{yurt2020semi,
  title={Semi-Supervised Learning of Mutually Accelerated MRI Synthesis without Fully-Sampled Ground Truths},
  author={Yurt, Mahmut and Hassan Dar, Salman Ul and {\"O}zbey, Muzaffer and T{\i}naz, Berk and Karl{\i} O{\u{g}}uz, Kader and {\c{C}}ukur, Tolga},
  journal={arXiv e-prints},
  pages={arXiv--2011},
  year={2020}
}
```
For any questions, comments and contributions, please contact Mahmut Yurt (myurt[at]stanford.edu) <br />

(c) ICON Lab 2022

## Acknowledgments
This code uses libraries from [pGAN](https://github.com/icon-lab/pGAN-cGAN) and [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository.
