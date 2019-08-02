import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
from torchvision.models import vgg16_bn


# this is a copy of:
#   https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres-imagenet.ipynb

torch.cuda.set_device(0)

path = Path('data')
path_input = Path('data/input')
path_bone = Path('data/bone')
path_tissue = Path('data/tissue')

#
# Resize tissue and input
#

path_input_128  = Path('data/input_128')
path_bone_128   = Path('data/bone_128')
path_tissue_128 = Path('data/tissue_128')

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
bs = 8,128
sets_input  = [(path_input_128,128)]
sets_tissue = [(path_tissue_128,128)]
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

free = gpu_mem_get_free_no_cache()
# the max size of the test image depends on the available GPU RAM 
if free > 8200: 
    bs,size=16,128
else:           
    bs,size=8,128
print(f"using bs={bs}, size={size}, have {free}MB of GPU RAM free")

arch = models.resnet34
# sample = 0.1
sample = False

tfms = get_transforms()

# we want to predict the tissue from the input
src = ImageImageList.from_folder(path_input_128)

if sample: 
    src = src.filter_by_rand(sample, seed=42)

src = src.split_by_rand_pct(0.1, seed=42)

# ok, we need to use the path_tissue_128 as the target for the input of path_input_128
def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_tissue_128/x.relative_to(path_input_128))
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

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
gc.collect();

learn.unfreeze()

# we would need to learn this first?
# learn.load((path_pets/'small-96'/'models'/'2b').absolute());

learn.fit_one_cycle(40, slice(1e-6,1e-4))
learn.recorder.plot_lr()
# learn.recorder.plot()

learn.save('deboning')

# show output
learn.show_results(rows=3, imgsize=5)


learn.recorder.plot_losses()

#
# Testing the network
#
_=learn.load('deboning')

data_mr = (ImageImageList.from_folder(path_input).split_by_rand_pct(0.1, seed=42)
          .label_from_func(lambda x: path_tissue_128/x.relative_to(path_input))
          .transform(get_transforms(), size=(820,1024), tfm_y=True)
          .databunch(bs=2).normalize(imagenet_stats, do_y=True))

learn.data = data_mr

fn = path_input/'img_0050.png'
img = PIL.Image.open(fn); img = img.convert('L');
fn = '/tmp/input.png'
img.save(fn)
img = open_image(fn); img.shape

_,img_hr,b = learn.predict(img)

show_image(img, figsize=(18,15), interpolation='nearest');
