import base64
import random
from io import BytesIO
import torch
from torchvision import models, transforms
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics.pairwise import cosine_similarity

def getBase64FromTestImage(image):
    img = image.resize((270, 480))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# 이미지에서 특징 벡터 추출 함수 정의
def extract_features(img_path, model):
    
    # 이미지 전처리
    preprocess = transforms.Compose([
        transforms.Resize(256),                    # 크기 조정
        transforms.CenterCrop(224),                # 224x224로 자르기
        transforms.ToTensor(),                     # Tensor로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    
    img = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)  # 배치 차원 추가
    
    # 2.2. 모델을 사용해 특징 벡터 추출 (특징 맵 추출)
    with torch.no_grad():  # 추론 시에는 gradients 계산을 하지 않음
        features = model(img_tensor)
    
    return features.flatten().numpy()

def find_faiss_similar_images(query_img_path, model, faiss_index, image_files, top_k):
    # 쿼리 이미지의 특징 벡터 추출
    query_features = extract_features(query_img_path, model).astype('float32').reshape(1, -1)
    # FAISS를 사용해 유사한 이미지 검색
    distances, indices = faiss_index.search(query_features, top_k)
    # 결과 출력
    similar_images = []
    for i in range(top_k):
        similar_images.append(image_files[indices[0][i]])
    return similar_images

# 4. 유사 이미지 검색 함수 (코사인 유사도 사용)
def find_similar_images(query_img_path, model, feature_list, image_files, top_k):
    similar_images = []
    
    # 쿼리 이미지의 특징 벡터 추출
    query_features = extract_features(query_img_path, model)
    
    # 모든 이미지의 특징 벡터와 코사인 유사도 계산
    similarities = cosine_similarity([query_features], feature_list)[0]
    
    # 유사도가 높은 이미지 순으로 정렬
    sorted_indices = np.argsort(similarities)[::-1]
    
    # 유사도가 높은 상위 top_k개 이미지 출력
    for i in range(top_k):
        # print(f"Similar Image {i+1}: {image_files[sorted_indices[i]]} (Similarity: {similarities[sorted_indices[i]]})")
        similar_images.append(image_files[sorted_indices[i]])    
    return similar_images

def apply_flipping(img):
    """Flip an image horizontally (left to right)."""
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def apply_rotation(img, angle=180):
    return img.rotate(angle, expand=True)

def apply_saturation(img, factor=1.5):
    """Apply saturation enhancement to an image."""
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)

def apply_brightness(img, factor=1.5):
    """Apply brightness enhancement to an image."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def apply_saliency(img):
    # PIL 이미지를 OpenCV 형식으로 변환
    image_cv = np.array(img.convert("RGB"))[:, :, ::-1]  # RGB to BGR

    # Saliency Map 생성
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image_cv)
    saliency_map = (saliency_map * 255).astype(np.uint8) if success else np.zeros(image_cv.shape[:2], dtype=np.uint8)

    # Edge Detection
    edges = cv2.Canny(image_cv, 100, 200)

    # Saliency와 Edge 결합
    combined = cv2.addWeighted(saliency_map, 0.7, edges, 0.3, 0)

    # 결과를 PIL 이미지로 변환
    return Image.fromarray(combined)

def apply_grayscale(img):
    """Convert an image to grayscale."""
    return ImageOps.grayscale(img)

# Fewshot 판다스 데이터 생성
def getRealAugmentedImageFewshot(serializer, top_similarities, augmented_img_num):
    html_list = []
    base64_list = []
    elements_list = []
    label_key='labels'
    bbox_key='discrete_gold_bboxes'
    train_image_path = r'C:\Users\didtn\LayoutGeneration\LayoutPrompter\LayoutGeneration-main\LayoutGeneration-main\LayoutPrompter\dataset\rico\original_img\trainjpg\\'
    transformations = [
        apply_saliency,
        apply_flipping,
        apply_brightness,
        apply_saturation,
        apply_rotation,
        apply_grayscale
    ]

    # 데이터 증강 필요없는 유사 이미지들
    for sample in top_similarities:
        sample_image = train_image_path + sample['name']
        image = Image.open(sample_image)
        html_output = serializer._build_html_output(sample, label_key, bbox_key)
        html_list.append(html_output)
        base64_list.append(getBase64FromTestImage(image))
        elements_list.append(serializer.build_input(sample))

    top_sample = top_similarities[0]
    top_sample_path = train_image_path + top_sample['name']
    top_image = Image.open(top_sample_path)
    top_image = top_image.resize((270, 480))

    top_html_output = serializer._build_html_output(sample, label_key, bbox_key)
    top_element_list = serializer.build_input(top_sample)
    
    #데이터 증강이 필요한 이미지들
    for index in range(augmented_img_num):
        transform = transformations[index]
        transformed_img = transform(top_image)
        base64_list.append(getBase64FromTestImage(transformed_img))
        if (transform == apply_rotation):
            rotate_html_output = serializer._build_rotate_html_output(sample, label_key, bbox_key, angle=180)
            html_list.append(rotate_html_output)
        elif (transform == apply_flipping):
            flip_html_output = serializer._build_flip_html_output(sample, label_key, bbox_key, flip_horizontal=True)
            html_list.append(flip_html_output)
        else:
            html_list.append(top_html_output)
        elements_list.append(top_element_list)

    # 데이터프레임 생성
    df = pd.DataFrame({
        'Element':elements_list,
        'HTML': html_list,
        'Base64': base64_list
    })
    return df

