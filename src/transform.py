from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from PIL import Image
import numpy as np

# Edge Augmentation Class
class EdgeAugmentation:
    def __init__(self):
        pass

    def apply_edge_detection(self, image):
        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply edge detection (e.g., Canny)
        edges = canny(image, sigma=1.0)
        return edges

    def apply_nms_thinning(self, edges):
        # Apply NMS to thin edges
        return skimage.morphology.thin(edges)

    def apply_thresholding(self, edges):
        # Apply Otsu thresholding as an example (you can randomize it)
        thresh_value = skimage.filters.threshold_otsu(edges)
        binary_image = edges > thresh_value
        return binary_image

    def __call__(self, image):
        edges = self.apply_edge_detection(image)
        thinned_edges = self.apply_nms_thinning(edges)
        final_image = self.apply_thresholding(thinned_edges)
        return final_image
        
class TorchvisionTransform:
    def __init__(self, is_train: bool = True):
        common_transforms = [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        
        if is_train:
            self.transform = transforms.Compose(
                [
                    AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
                ] + common_transforms
            )
        else:
            self.transform = transforms.Compose(common_transforms)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        common_transforms = [
            A.Resize(448, 448),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        if is_train:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15),
                    A.RandomBrightnessContrast(p=0.2),
                ] + common_transforms
            )
        else:
            self.transform = A.Compose(common_transforms)

    def __call__(self, image) -> torch.Tensor:
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        transformed = self.transform(image=image)
        return transformed['image']


class TransformSelector:
    def __init__(self, transform_type: str):
        if transform_type in ["torchvision", "albumentations"]:
            self.transform_type = transform_type
        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool):
        if self.transform_type == 'torchvision':
            return TorchvisionTransform(is_train=is_train)
        elif self.transform_type == 'albumentations':
            return AlbumentationsTransform(is_train=is_train)
