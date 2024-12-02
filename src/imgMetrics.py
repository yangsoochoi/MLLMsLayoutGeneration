from PIL import Image
import numpy as np
from torchvision import models, transforms
import torch
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim


def calculate_ssim(imageA, imageB):
    # 이미지를 NumPy 배열로 변환합니다.
    np_imageA = np.array(imageA)
    np_imageB = np.array(imageB)
    
    # 이미지를 그레이스케일로 변환합니다.
    np_imageA_gray = np_imageA if len(np_imageA.shape) == 2 else np.dot(np_imageA[...,:3], [0.2989, 0.5870, 0.1140])
    np_imageB_gray = np_imageB if len(np_imageB.shape) == 2 else np.dot(np_imageB[...,:3], [0.2989, 0.5870, 0.1140])

    # SSIM 계산 (data_range를 명시적으로 지정합니다.)
    score, _ = ssim(np_imageA_gray, np_imageB_gray, full=True, data_range=np_imageA_gray.max() - np_imageA_gray.min())
    return score

# Function to calculate FID between two PIL images
def calculate_fid(image1, image2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.fc = torch.nn.Identity()  # Remove the final fully connected layer

    mu1, sigma1 = calculate_activation_statistics([image1], inception_model, device)
    mu2, sigma2 = calculate_activation_statistics([image2], inception_model, device)

    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid

# Helper function to calculate the mean and covariance matrix of the activations
def calculate_activation_statistics(images, model, device):
    model.eval()
    with torch.no_grad():
        activations = []
        for image in images:
            image = image.convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = transform(image).unsqueeze(0).to(device)
            activation = model(image).cpu().numpy().reshape(-1)
            activations.append(activation)
        activations = np.array(activations)
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma
    
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # sigma1과 sigma2가 올바른 형태의 행렬인지 확인
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    # 공분산 행렬의 곱의 제곱근을 계산
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    # 만약 covmean이 비정규 행렬이거나 NaN이 포함된 경우, eps를 추가하여 수치를 안정화
    if not np.isfinite(covmean).all():
        covmean = sqrtm((sigma1 + np.eye(sigma1.shape[0]) * eps).dot(sigma2 + np.eye(sigma2.shape[0]) * eps))
    # 차이 벡터 계산
    diff = mu1 - mu2
    # Frechet Distance 계산
    fid = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)    
    return fid

from utils import compute_overlap, compute_alignment, convert_ltwh_to_ltrb
    
def calculateOverlap(response):
    # 예측된 데이터셋의 라벨 및 바운딩 박스 리스트
    pred_labels = response[0]  # response[0]은 라벨 텐서
    pred_bboxes = response[1]  # response[1]은 바운딩 박스 텐서

    _pred_labels = pred_labels.unsqueeze(0)
    _pred_bboxes = convert_ltwh_to_ltrb(pred_bboxes).unsqueeze(0)
    _pred_padding_mask = torch.ones_like(_pred_labels).bool()
    score = compute_overlap(_pred_bboxes, _pred_padding_mask)
    return score

def calculateAllignment(response):
    # 예측된 데이터셋의 라벨 및 바운딩 박스 리스트
    pred_labels = response[0]  # response[0]은 라벨 텐서
    pred_bboxes = response[1]  # response[1]은 바운딩 박스 텐서

    _pred_labels = pred_labels.unsqueeze(0)
    _pred_bboxes = convert_ltwh_to_ltrb(pred_bboxes).unsqueeze(0)
    _pred_padding_mask = torch.ones_like(_pred_labels).bool()
    score = compute_alignment(_pred_bboxes, _pred_padding_mask)
    return score