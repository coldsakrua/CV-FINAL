# Best Results

## 1. Denoising

### 1.1 Manually Stopped
```
python main.py --task=denoise --name=f16 --epoch=3000
python main.py --task=denoise --name=snail --epoch=3000
python main.py --task=denoise --name=denoise_img1 --epoch=3000
python main.py --task=denoise --name=denoise_img2 --epoch=1800
```

### 1.2 Automatically Stopped  
You can set --auto as True to achieve automatically Stop  
```
python main.py --task=denoise --name=snail --epoch=4000 --auto=True
```

## 2. Inpainting

### 2.1 Text Inpainting
```
python main.py --task=inpaint --name=kate --epoch=10000
python main.py --task=inpaint --name=inpainting_img3 --epoch=5000
python main.py --task=inpaint --name=inpainting_img4 --epoch=10000
```
### 2.2 Region Inpainting
```
python main.py --task=inpaint --name=vase --epoch=20000
python main.py --task=inpaint --name=library --epoch=5000
python main.py --task=inpaint --name=inpainting_img1 --epoch=15000
python main.py --task=inpaint --name=inpainting_img2 --epoch=4000
```
## 3. Super Resolution
```
python main.py --task=super --name=zebra --epoch=2000
python main.py --task=super --name=man_sr --epoch=2000
python main.py --task=super --name=boat_sr --epoch=2000
python main.py --task=super --name=peppers_sr --epoch=2000
python main.py --task=super --name=montage_sr --epoch=2000
```