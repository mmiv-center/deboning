import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
from torchvision.models import vgg16_bn
# Problem with visual studio code at beginning:
#   > source ~/.bashrc
# first and the switch to the setup will work.


# This is largely a copy of source code found on:
#   https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres-imagenet.ipynb

defaults.cmap = 'binary'
CPU=1
if CPU == 1:
    defaults.device = torch.device("cpu")
else:
    # default is GPU
    torch.cuda.set_device(0)

path = Path('data_vessel_plus')
path_input = Path('data_vessel_plus/input')
#path_bone = Path('data_vessel_plus/bone')
path_tissue = Path('data_vessel_plus/target')

#
# Resize tissue and input
# (checks for the directory and only creates the output if the directory does not exist)
#

path_input_512  = Path('data/input_512')
path_bone_512   = Path('data/bone_512')
path_tissue_512 = Path('data/tissue_512')

il = ImageList.from_folder(path_input)
tl = ImageList.from_folder(path_tissue)

def resize_one_tissue(fn, i, path, size):
    dest = path/fn.relative_to(path_tissue)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn)
    targ_sz = resize_to(img, size, use_min=True)
    img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('L')
    img.save(dest, quality=60)

def resize_one_input(fn, i, path, size):
    dest = path/fn.relative_to(path_input)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn)
    targ_sz = resize_to(img, size, use_min=True)
    img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('L')
    img.save(dest, quality=60)

# create smaller versions of the images
bs = 8,512
sets_input  = [(path_input_512,512)]
sets_tissue = [(path_tissue_512,512)]
for p,size in sets_input:
    if not p.exists():
        print(f"resizing to {size} into {p}")
        parallel(partial(resize_one_input, path=p, size=size), il.items)

for p,size in sets_tissue:
    if not p.exists():
        print(f"resizing to {size} into {p}")
        parallel(partial(resize_one_tissue, path=p, size=size), tl.items)

#
# Resize DONE
#

bs,size=6,1024

if not(CPU == 1):
    free = gpu_mem_get_free_no_cache()
    # the max size of the test image depends on the available GPU RAM 
    if free > 8200: 
        bs,size=16,128
    else:           
        bs,size=8,128
    print(f"using bs={bs}, size={size}, have {free}MB of GPU RAM free")

#arch = models.resnet34
arch = models.resnet50
# sample = 0.1
sample = False

#tfms = get_transforms()
tfms = get_transforms(max_rotate=5, max_zoom=1.3, max_lighting=0.4, max_warp=0.1,
                      p_affine=1., p_lighting=1.)

# we want to predict the tissue from the input
src = ImageImageList.from_folder(path_input)

if sample: 
    src = src.filter_by_rand(sample, seed=42)

src = src.split_by_rand_pct(0.1, seed=42)

# ok, we need to use the path_tissue_128 as the target for the input of path_input_128
def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_tissue/x.relative_to(path_input))
           .transform(tfms, size=size, tfm_y=True)
           .databunch(bs=bs).normalize(do_y=True))

    data.c = 3
    return data

data = get_data(bs,size)

#
# Feature list
#

def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)

if CPU == 1:
    vgg_m = vgg16_bn(True).features.eval()
else:
    vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)
blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]

base_loss = F.l1_loss

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()

feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])

#
# Train the network
#

wd = 1e-3
learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics, blur=True, norm_type=NormType.Weight)
gc.collect()

learn.unfreeze()

# we would need to learn this first?
# learn.load((path_pets/'small-96'/'models'/'2b').absolute());

# find a good learning rate:
# learn.lr_find()
# learn.recorder.plot(skip_end=0)

learn.fit_one_cycle(60, slice(1e-6,1e-4))
learn.recorder.plot_lr()
# learn.recorder.plot(skip_end=50)

learn.save('../../../models/deboning_1024_60steps_resnet50')

# show output
learn.show_results(rows=5, imgsize=10)


learn.recorder.plot_losses()


def my_show_results(self, ds_type=DatasetType.Valid, rows:int=5, **kwargs):
    "Show `rows` result of predictions on `ds_type` dataset."
    #TODO: get read of has_arg x and split_kwargs_by_func if possible
    #TODO: simplify this and refactor with pred_batch(...reconstruct=True)
    n_items = rows ** 2 if self.data.train_ds.x._square_show_res else rows
    if self.dl(ds_type).batch_size < n_items: n_items = self.dl(ds_type).batch_size
    ds = self.dl(ds_type).dataset
    self.callbacks.append(RecordOnCPU())
    preds = self.pred_batch(ds_type)
    *self.callbacks,rec_cpu = self.callbacks
    x,y = rec_cpu.input,rec_cpu.target
    norm = getattr(self.data,'norm',False)
    if norm:
        x = self.data.denorm(x)
        if norm.keywords.get('do_y',False):
            y     = self.data.denorm(y, do_x=True)
            preds = self.data.denorm(preds, do_x=True)
    analyze_kwargs,kwargs = split_kwargs_by_func(kwargs, ds.y.analyze_pred)
    preds = [ds.y.analyze_pred(grab_idx(preds, i), **analyze_kwargs) for i in range(n_items)]
    xs = [ds.x.reconstruct(grab_idx(x, i)) for i in range(n_items)]
    if has_arg(ds.y.reconstruct, 'x'):
        ys = [ds.y.reconstruct(grab_idx(y, i), x=x) for i,x in enumerate(xs)]
        zs = [ds.y.reconstruct(z, x=x) for z,x in zip(preds,xs)]
    else :
        ys = [ds.y.reconstruct(grab_idx(y, i)) for i in range(n_items)]
        zs = [ds.y.reconstruct(z) for z in preds]
    my_show_xyzs(ds.x, xs, ys, zs, **kwargs)

import numpy
import torchvision
def my_show_xyzs(self, xs, ys, zs, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
    "Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`."
    title = 'Input / Prediction / Target / Input - Target'
    axs = subplots(len(xs), 4, imgsize=imgsize, figsize=figsize, title=title, weight='bold', size=14)
    for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
        x.show(ax=axs[i,0], **kwargs)
        y.show(ax=axs[i,2], **kwargs)
        z.show(ax=axs[i,1], **kwargs)
        s_image = ((x.data[0,:,:]/255.0 - z.data[0,:,:]/255.0)+0.1)*255
        s = Image(pil2tensor(torchvision.transforms.ToPILImage()(s_image), np.float32))
        s.show(ax=axs[i,3], **kwargs)


my_show_results(learn, rows=5, imgsize=10)

#
# Testing the network
# (use higher resolution data and see if that works still)
defaults.device = 'cpu'
_=learn.load('../../../models/deboning_1024_60steps')

path_input  = Path('data/input')
path_tissue = Path('data/tissue')

# This one is not needed, we can use predict immediately.
data_mr = (ImageImageList.from_folder(path_input).split_by_rand_pct(0.1, seed=42)
          .label_from_func(lambda x: path_tissue/x.relative_to(path_input))
          .transform(get_transforms(), size=(1024,1024), tfm_y=True)
          .databunch(bs=2).normalize(do_y=True))#

learn.data = data_mr

fn = path_input/'img_0050.png'
fn = 'x-ray/0B5408Fd04Ec461B8E4BC0B52D8D0103.png'
fn = 'x-ray/5D2Ea647162E4E4EB894E97F1F22871D.png'
fn = 'x-ray/7B529D8FCf4443Ea8742692C95Cf2C2C.png'
fn = 'x-ray/9D47938A5Fd6450696FaF15E1B9F55E6.png'
fn = 'x-ray/20D2C1CeB41C4A7CB951Afa98Be08853.png'
fn = 'x-ray/43025118F9C547F2Be9AEfc8C335Ba33.png'
fn = 'x-ray/B0Fc8Bfb89Ad49C3A015Df227D5D6038.png'

img = PIL.Image.open(fn); img = img.convert('L').resize([1024,1024]);
fn = '/tmp/input.png'
img.save(fn)
img = open_image(fn); img.shape

_,img_hr,b = learn.predict(img)

show_image(img, figsize=(18,15), interpolation='nearest')
show_image(img_hr, figsize=(18,15), interpolation='nearest')
s_image = ((img.data[0,:,:]/255.0 - img_hr.data[0,:,:]/255.0)+0.1)*255
s = Image(pil2tensor(torchvision.transforms.ToPILImage()(s_image), np.float32))
show_image(s, figsize=(18,15), interpolation='nearest')
