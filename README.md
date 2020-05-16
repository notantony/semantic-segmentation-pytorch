# Parent repository
[Covers 2019](https://github.com/notantony/covers2019) 

# API

[Classes table](https://docs.google.com/spreadsheets/d/1QXMxMCEpFli6m4f9IZxZiyCgDPId5fGCN_pHVDVeNPc/edit#gid=0)

## Single object cropping
Crop segment of requested class. Border extention and Gaussian blur can be applied for smoothing borders.

#### Input:
Address: `/segmentation`, POST \
MIMEs: `applcation/json`

Parameters: \
`data`: base64-encoded image. \
`name`: requested class name. \
`type`: optional, image extension. Can be omitted for most of extensions. \
`blur_radius`: optional, blur radius. \
`border_extension`: optional, amount of pixels to extend the border.

#### Output:
JSON: \
`image`: base64-encoded resulting `.png` file. \
If image is missing required class, `400` request code will be returned.

<details>
  <summary> <b>Sample: </b> </summary> 

  Request JSON:
  ```json
  {
      "name" : "person",
      "type" : "jpeg",
      "data" : "/9j/4AAQSkZJRgABAQEASABIAAD//gATQ3JlYXRlZCB3a..."
  }
  ```
  
  Response:
  ```json
  {
      "image" : "iVBORw0KGgoAAAANSUhEUgAAA+EAAAI1CAYAAA..."
  }
  ```
</details>


## Segmentation map constructing
Run segmentation on requested image.  

#### Input:
Address: `/colormap`, POST \
MIMEs: `applcation/json`, `image/jpeg`, `image/png`

Parameters when Json: \
`data`: base64-encoded image. \
`type`: optional, image extension. Can be omitted for most of extensions.

#### Output:
JSON: \
`colormap`: base64-encoded bytes representation of numpy array containing classes of each pixel. Serialized with [numpy.tobytes()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tobytes.html), can be desrialized with [numpy.frombuffer()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.frombuffer.html). \
`shape`: shape of the array, equals to original image shape. \
`dtype`: dtype of the array, currently always equals to `int16`.

<details>
  <summary> <b>Sample: </b> </summary> 

  Request JSON:
  ```json
  {
      "type" : "jpeg",
      "data" : "/9j/4AAQSkZJRgABAQEASABIAAD//gATQ3JlYXRlZCB3a..."
  }
  ```
  
  Response:
  ```json
  {
      "colormap": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA...",
      "dtype": "int16",
      "shape": "(588, 1000)"
  }
  ```
</details>

# Running

Run with `run.sh`:
```
chmod +x run.sh
./run.sh
```

Encode images into base64 with `b64.sh` (creates `tmp.txt`):
```
chmod +x b64.sh
./b64.sh <image_path>
```

# Reference

See [origin repository](https://github.com/CSAILVision/semantic-segmentation-pytorch) for more info.

Semantic Understanding of Scenes through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso and A. Torralba. International Journal on Computer Vision (IJCV), 2018. (https://arxiv.org/pdf/1608.05442.pdf)

    @article{zhou2018semantic,
      title={Semantic understanding of scenes through the ade20k dataset},
      author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Xiao, Tete and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
      journal={International Journal on Computer Vision},
      year={2018}
    }

Scene Parsing through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. (http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf)

    @inproceedings{zhou2017scene,
        title={Scene Parsing through ADE20K Dataset},
        author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2017}
    }
    
